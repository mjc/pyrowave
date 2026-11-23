// Copyright (c) 2025 Hans-Kristian Arntzen
// SPDX-License-Identifier: MIT
#include "pyrowave_decoder.hpp"
#include "device.hpp"
#include "buffer.hpp"
#include "image.hpp"
#include "math.hpp"
#include "pyrowave_common.hpp"
#include <algorithm>

namespace PyroWave
{
using namespace Granite;
using namespace Vulkan;

struct DequantizerPushData
{
	ivec2 resolution;
	int32_t output_layer;
	int32_t block_offset_32x32;
	int32_t block_stride_32x32;
	int32_t band_stride_32x32;
};

static bool select_uma_buffer_domain(const Device &dev, BufferDomain &out_domain)
{
	const auto &mem_props = dev.get_memory_properties();
	constexpr VkMemoryPropertyFlags cached_flags =
		VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
		VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
	constexpr VkMemoryPropertyFlags coherent_flags =
		VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
		VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
	constexpr VkDeviceSize min_uma_heap_size = 512ull * 1024ull * 1024ull;

	bool has_cached = false;
	bool has_coherent = false;

	for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++)
	{
		const auto &type = mem_props.memoryTypes[i];
		if (type.heapIndex >= mem_props.memoryHeapCount ||
		    mem_props.memoryHeaps[type.heapIndex].size < min_uma_heap_size)
			continue;

		if ((type.propertyFlags & cached_flags) == cached_flags)
			has_cached = true;
		if ((type.propertyFlags & coherent_flags) == coherent_flags)
			has_coherent = true;
	}

	if (has_cached)
	{
		// Prefer CPU-cached UMA memory (coherent when available, otherwise flushed via unmap_host_buffer).
		out_domain = BufferDomain::UMACachedCoherentPreferDevice;
		return true;
	}
	if (has_coherent)
	{
		out_domain = BufferDomain::LinkedDeviceHost;
		return true;
	}
	return false;
}

struct Decoder::Impl final : public WaveletBuffers
{
	static constexpr uint32_t kUmaRingSlots = 4;

	BufferHandle dequant_offset_buffer, payload_data, edge_wg_list_buffer, active_blocks_buffer;
	BufferViewHandle payload_u32_view, payload_u16_view, payload_u8_view;

	// Ring-buffered zero-copy UMA upload buffers (used when !use_readonly_texel_buffer and UMA heap is present).
	bool use_uma_upload = false;
	bool uma_offset_direct = false;
	bool uma_offset_synced_to_cpu = false;
	BufferDomain uma_domain = BufferDomain::LinkedDeviceHost;
	uint32_t uma_slot = 0;
	BufferHandle uma_dequant_offset[kUmaRingSlots];
	BufferHandle uma_payload_data[kUmaRingSlots];
	BufferHandle uma_active_blocks[kUmaRingSlots];
	uint32_t *uma_payload_write_ptr = nullptr;
	uint32_t *uma_offset_write_ptr = nullptr;
	size_t uma_payload_written_words = 0;
	size_t uma_payload_cap_words = 0;
	bool uma_slot_submitted = false;
	uint32_t uma_submitted_slot = 0;
	VkDeviceSize uma_submitted_bytes = 0;

	// Turbo-hacky path.
	DeviceAllocationOwnerHandle linear_memory;
	ImageHandle payload_r8_image, payload_r16_image, payload_r32_image;
	bool need_image_transition = true;

	std::vector<uint32_t> dequant_offset_buffer_cpu;
	std::vector<uint32_t> payload_data_cpu;
	// Per-block lookup packed as (level << 4) | (component << 2) | band for O(1) active tracking.
	std::vector<uint8_t> block_to_comp_level_band;
	// Per-block packed (bx | (by << 12) | (band << 24)) for packed 1D active-block dequant dispatch.
	std::vector<uint32_t> block_dequant_coord;
	struct ActiveBlockEntry
	{
		uint32_t coord;
		uint32_t offset;
	};
	std::vector<ActiveBlockEntry> active_block_list[NumComponents][DecompositionLevels];
	uint32_t active_block_offset[NumComponents][DecompositionLevels] = {};
	uint32_t active_hf_blocks[NumComponents][DecompositionLevels] = {};
	struct IdwtLevelLayout
	{
		int32_t ix_end = 0;
		int32_t iy_end = 0;
		uint32_t edge_offset = 0;
		uint32_t edge_count = 0;
		bool use_split = false;
	};
	IdwtLevelLayout idwt_level_layouts[DecompositionLevels] = {};
	std::vector<uint32_t> edge_wg_list_cpu;
	bool edge_wg_list_uploaded = false;
	bool subgroup_ballot_mask = false;
	bool prewarmed_idwt_intermediate = false;
	bool prewarmed_idwt_final = false;
	float prewarmed_range_scale = 0.0f;
	int decoded_blocks = 0;
	int total_blocks_in_sequence = 0;
	uint32_t last_seq = UINT32_MAX;
	bool decoded_frame_for_current_sequence = false;

	bool push_packet(const void *data, size_t size);
	bool decode(CommandBuffer &cmd, const ViewBuffers &views);
	bool decode_is_ready(bool allow_partial_frame) const;
	bool decode_is_ready(bool allow_partial_frame, int pristine_bands, float received_ratio,
	                     const uint32_t *active_block_mask, size_t word_count) const;
	bool has_pristine_bands(int bands, const uint32_t *active_block_mask, size_t word_count) const;


	bool dequant(CommandBuffer &cmd);
	bool idwt(CommandBuffer &cmd, const ViewBuffers &views);
	bool idwt_fragment(CommandBuffer &cmd, const ViewBuffers &views);
	void init_block_meta() override;
	void clear();

	void upload_payload(CommandBuffer &cmd);
	bool upload_payload_uma(CommandBuffer &cmd);
	void begin_uma_payload_write();
	void discard_uma_payload_write();
	void build_static_edge_wg_lists();

	void check_linear_texture_support();
};

Decoder::Decoder()
{
	impl.reset(new Impl);
}

Decoder::~Decoder()
{
	if (impl)
		impl->discard_uma_payload_write();
}

void Decoder::Impl::upload_payload(CommandBuffer &cmd)
{
	VkDeviceSize required_size = payload_data_cpu.size() * sizeof(uint32_t);

	// Avoid edge case OOB access without robustness on the payload buffer during dequant.
	VkDeviceSize required_size_padded = required_size + 16;

	if (!payload_data || required_size_padded > payload_data->get_create_info().size)
	{
		BufferCreateInfo bufinfo;
		bufinfo.size = std::max<VkDeviceSize>(64 * 1024, required_size_padded * 2);
		bufinfo.usage =
			VK_BUFFER_USAGE_TRANSFER_DST_BIT |
			VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
			VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT;
		bufinfo.domain = BufferDomain::Device;
		payload_data = device->create_buffer(bufinfo);
		device->set_name(*payload_data, "payload-data");

		if (use_readonly_texel_buffer)
		{
			BufferViewCreateInfo view_info = {};
			view_info.buffer = payload_data.get();
			view_info.range = VK_WHOLE_SIZE;

			view_info.format = VK_FORMAT_R8_UINT;
			payload_u8_view = device->create_buffer_view(view_info);
			view_info.format = VK_FORMAT_R16_UINT;
			payload_u16_view = device->create_buffer_view(view_info);
			view_info.format = VK_FORMAT_R32_UINT;
			payload_u32_view = device->create_buffer_view(view_info);
		}

		// This shouldn't happen to demote to texel buffers if we need to deal with massive gigantic payloads.
		payload_r8_image.reset();
		payload_r16_image.reset();
		payload_r32_image.reset();
	}

	if (need_image_transition)
	{
		cmd.begin_barrier_batch();
		const Image *imgs[] = { payload_r8_image.get(), payload_r16_image.get(), payload_r32_image.get() };
		for (auto *img : imgs)
		{
			if (img)
			{
				cmd.image_barrier(*img, VK_IMAGE_LAYOUT_PREINITIALIZED, VK_IMAGE_LAYOUT_GENERAL,
				                  0, 0, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
			}
		}
		cmd.end_barrier_batch();
		need_image_transition = false;
	}

	if (!payload_data_cpu.empty())
		memcpy(cmd.update_buffer(*payload_data, 0, required_size), payload_data_cpu.data(), required_size);
}

void Decoder::Impl::discard_uma_payload_write()
{
	if (uma_payload_write_ptr && uma_payload_data[uma_slot])
	{
		device->unmap_host_buffer(*uma_payload_data[uma_slot], MEMORY_ACCESS_WRITE_BIT);
		uma_payload_write_ptr = nullptr;
	}
	if (uma_offset_write_ptr && uma_dequant_offset[uma_slot])
	{
		device->unmap_host_buffer(*uma_dequant_offset[uma_slot], MEMORY_ACCESS_WRITE_BIT);
		uma_offset_write_ptr = nullptr;
	}
	uma_payload_written_words = 0;
	uma_payload_cap_words = 0;
}

void Decoder::Impl::begin_uma_payload_write()
{
	if (uma_slot_submitted)
	{
		uma_slot = (uma_slot + 1) % kUmaRingSlots;
		uma_slot_submitted = false;
		uma_submitted_bytes = 0;
	}

	uma_payload_written_words = 0;
	uma_payload_cap_words = 0;
	if (!use_uma_upload)
		return;

	if (uma_offset_direct && !uma_offset_write_ptr && uma_dequant_offset[uma_slot])
	{
		uma_offset_write_ptr = static_cast<uint32_t *>(device->map_host_buffer(
			*uma_dequant_offset[uma_slot], MEMORY_ACCESS_READ_BIT | MEMORY_ACCESS_WRITE_BIT));
	}

	if (uma_payload_write_ptr && uma_payload_data[uma_slot])
	{
		uma_payload_cap_words = size_t(uma_payload_data[uma_slot]->get_create_info().size / sizeof(uint32_t));
		return;
	}

	auto &payload_buf = uma_payload_data[uma_slot];
	if (!payload_buf)
	{
		BufferCreateInfo bufinfo;
		// Pre-size ring slots to hold typical frames without mid-frame reallocations.
		VkDeviceSize initial_bytes = std::max<VkDeviceSize>(256 * 1024, VkDeviceSize(aligned_width) * VkDeviceSize(aligned_height) / 2);
		bufinfo.size = initial_bytes;
		bufinfo.usage =
			VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
			VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT |
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
			VK_BUFFER_USAGE_TRANSFER_DST_BIT;
		bufinfo.domain = uma_domain;
		payload_buf = device->create_buffer(bufinfo);
		if (!payload_buf)
			return;
		device->set_name(*payload_buf, "payload-data-uma");
	}

	uma_payload_write_ptr = static_cast<uint32_t *>(device->map_host_buffer(*payload_buf, MEMORY_ACCESS_WRITE_BIT));
	if (uma_payload_write_ptr)
		uma_payload_cap_words = size_t(payload_buf->get_create_info().size / sizeof(uint32_t));
}

bool Decoder::Impl::upload_payload_uma(CommandBuffer &cmd)
{
	VkDeviceSize tail_size = payload_data_cpu.size() * sizeof(uint32_t);

	if (uma_slot_submitted)
	{
		// Incremental decode of additional packets on an already-submitted sequence.
		uint32_t prev_slot = uma_submitted_slot;
		VkDeviceSize prev_bytes = uma_submitted_bytes;
		VkDeviceSize total_size = prev_bytes + tail_size;
		VkDeviceSize total_padded = total_size + 16;

		if (tail_size == 0 && uma_payload_data[prev_slot])
		{
			uma_slot = prev_slot;
			return true;
		}
		else
		{
			uma_slot = (uma_slot + 1) % kUmaRingSlots;
			auto &dst_buf = uma_payload_data[uma_slot];
			if (!dst_buf || total_padded > dst_buf->get_create_info().size)
			{
				BufferCreateInfo bufinfo;
				bufinfo.size = std::max<VkDeviceSize>(64 * 1024, total_padded * 2);
				bufinfo.usage =
					VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
					VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT |
					VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
					VK_BUFFER_USAGE_TRANSFER_DST_BIT;
				bufinfo.domain = uma_domain;
				dst_buf = device->create_buffer(bufinfo);
				if (!dst_buf)
					return false;
				device->set_name(*dst_buf, "payload-data-uma");
			}
			if (prev_bytes > 0 && uma_payload_data[prev_slot])
				cmd.copy_buffer(*dst_buf, 0, *uma_payload_data[prev_slot], 0, prev_bytes);
			if (tail_size > 0)
				memcpy(cmd.update_buffer(*dst_buf, prev_bytes, tail_size), payload_data_cpu.data(), tail_size);
			memset(cmd.update_buffer(*dst_buf, total_size, 16), 0, 16);
			cmd.barrier(VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
			            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT);

			uma_submitted_slot = uma_slot;
			uma_submitted_bytes = total_size;
			uma_payload_written_words = size_t(total_size / sizeof(uint32_t));
			payload_data_cpu.clear();
		}
	}
	else
	{
		VkDeviceSize direct_size = uma_payload_written_words * sizeof(uint32_t);
		VkDeviceSize required_size = direct_size + tail_size;
		VkDeviceSize required_size_padded = required_size + 16;

		auto &payload_buf = uma_payload_data[uma_slot];
		if (!payload_buf || required_size_padded > payload_buf->get_create_info().size)
		{
			BufferCreateInfo bufinfo;
			bufinfo.size = std::max<VkDeviceSize>(64 * 1024, required_size_padded * 2);
			bufinfo.usage =
				VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
				VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT |
				VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
				VK_BUFFER_USAGE_TRANSFER_DST_BIT;
			bufinfo.domain = uma_domain;
			BufferHandle new_buf = device->create_buffer(bufinfo);
			if (!new_buf)
				return false;
			device->set_name(*new_buf, "payload-data-uma");

			if (uma_payload_write_ptr && payload_buf)
			{
				if (direct_size > 0)
					device->unmap_host_buffer(*payload_buf, MEMORY_ACCESS_WRITE_BIT, 0, direct_size);
				uma_payload_write_ptr = nullptr;
			}

			if (direct_size > 0 && payload_buf)
				cmd.copy_buffer(*new_buf, 0, *payload_buf, 0, direct_size);
			if (tail_size > 0)
				memcpy(cmd.update_buffer(*new_buf, direct_size, tail_size), payload_data_cpu.data(), tail_size);
			memset(cmd.update_buffer(*new_buf, required_size, 16), 0, 16);
			cmd.barrier(VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
			            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT);

			payload_buf = std::move(new_buf);
		}
		else if (uma_payload_write_ptr)
		{
			if (tail_size > 0)
				memcpy(reinterpret_cast<uint8_t *>(uma_payload_write_ptr) + direct_size, payload_data_cpu.data(), tail_size);
			memset(reinterpret_cast<uint8_t *>(uma_payload_write_ptr) + required_size, 0, 16);
			device->unmap_host_buffer(*payload_buf, MEMORY_ACCESS_WRITE_BIT, 0, required_size_padded);
			uma_payload_write_ptr = nullptr;
		}
		else
		{
			auto *payload_dst = static_cast<uint8_t *>(device->map_host_buffer(*payload_buf, MEMORY_ACCESS_WRITE_BIT));
			if (!payload_dst)
				return false;
			if (tail_size > 0)
				memcpy(payload_dst, payload_data_cpu.data(), tail_size);
			memset(payload_dst + tail_size, 0, 16);
			device->unmap_host_buffer(*payload_buf, MEMORY_ACCESS_WRITE_BIT, 0, required_size_padded);
		}

		uma_slot_submitted = true;
		uma_submitted_slot = uma_slot;
		uma_submitted_bytes = required_size;
		uma_payload_written_words = size_t(required_size / sizeof(uint32_t));
		uma_payload_cap_words = 0;
		payload_data_cpu.clear();
	}

	auto &offset_buf = uma_dequant_offset[uma_slot];
	if (!offset_buf)
		return false;

	VkDeviceSize offset_size = VkDeviceSize(block_count_32x32) * sizeof(uint32_t);
	if (offset_size > 0)
	{
		if (uma_offset_write_ptr)
		{
			device->unmap_host_buffer(*offset_buf, MEMORY_ACCESS_WRITE_BIT, 0, offset_size);
			uma_offset_write_ptr = nullptr;
		}
		else
		{
			auto *offset_dst = static_cast<uint8_t *>(device->map_host_buffer(*offset_buf, MEMORY_ACCESS_WRITE_BIT));
			if (!offset_dst)
				return false;
			memcpy(offset_dst, dequant_offset_buffer_cpu.data(), offset_size);
			device->unmap_host_buffer(*offset_buf, MEMORY_ACCESS_WRITE_BIT, 0, offset_size);
		}
	}

	if (packed_2x2_wavelets && uma_active_blocks[uma_slot])
	{
		uint32_t total_entries = 0;
		for (int l = 0; l < DecompositionLevels - 1; l++)
			for (int c = 0; c < NumComponents; c++)
				total_entries += active_hf_blocks[c][l];

		if (total_entries > 0)
		{
			auto *dst = static_cast<ActiveBlockEntry *>(device->map_host_buffer(*uma_active_blocks[uma_slot], MEMORY_ACCESS_WRITE_BIT));
			if (!dst)
				return false;
			uint32_t entry_offset = 0;
			for (int l = 0; l < DecompositionLevels - 1; l++)
			{
				for (int c = 0; c < NumComponents; c++)
				{
					active_block_offset[c][l] = entry_offset;
					const uint32_t count = active_hf_blocks[c][l];
					if (count > 0)
					{
						memcpy(dst + entry_offset, active_block_list[c][l].data(), count * sizeof(ActiveBlockEntry));
						entry_offset += count;
					}
				}
			}
			device->unmap_host_buffer(*uma_active_blocks[uma_slot], MEMORY_ACCESS_WRITE_BIT,
			                          0, VkDeviceSize(total_entries) * sizeof(ActiveBlockEntry));
		}
	}

	return true;
}

void Decoder::Impl::build_static_edge_wg_lists()
{
	edge_wg_list_cpu.clear();
	edge_wg_list_uploaded = false;
	std::fill(std::begin(idwt_level_layouts), std::end(idwt_level_layouts), IdwtLevelLayout{});
	if (!packed_2x2_wavelets)
		return;

	for (int level = 0; level < DecompositionLevels - 1; level++)
	{
		const int res_x = int(get_level_height(level));
		const int res_y = int(get_level_width(level));
		if (res_x < 34 || res_y < 34)
			continue;

		const int nx = (res_x + 15) / 16;
		const int ny = (res_y + 15) / 16;
		const int ix_end = std::min<int>(nx, (res_x - 18) / 16 + 1);
		const int iy_end = std::min<int>(ny, (res_y - 18) / 16 + 1);
		if (ix_end <= 1 || iy_end <= 1)
			continue;

		const int interior_wgs = (ix_end - 1) * (iy_end - 1);
		// Only split large levels (L0 and L1 on 1080p/4K); small levels like L2 run faster as one direct dispatch.
		if (interior_wgs < 1024)
			continue;

		auto &layout = idwt_level_layouts[level];
		layout.use_split = true;
		layout.ix_end = int32_t(ix_end);
		layout.iy_end = int32_t(iy_end);
		layout.edge_offset = uint32_t(edge_wg_list_cpu.size());

		const int edge_total = nx * ny - interior_wgs;
		edge_wg_list_cpu.resize(edge_wg_list_cpu.size() + size_t(edge_total));
		uint32_t *edst = edge_wg_list_cpu.data() + layout.edge_offset;
		for (int gy = 0; gy < ny; gy++)
			*edst++ = uint32_t(gy) << 16; // gx = 0
		for (int gx = ix_end; gx < nx; gx++)
		{
			const uint32_t gx_bits = uint32_t(gx);
			for (int gy = 0; gy < ny; gy++)
				*edst++ = gx_bits | (uint32_t(gy) << 16);
		}
		for (int gx = 1; gx < ix_end; gx++)
			*edst++ = uint32_t(gx); // gy = 0
		for (int gx = 1; gx < ix_end; gx++)
		{
			const uint32_t gx_bits = uint32_t(gx);
			for (int gy = iy_end; gy < ny; gy++)
				*edst++ = gx_bits | (uint32_t(gy) << 16);
		}
		layout.edge_count = uint32_t(edge_total);
	}
}

bool Decoder::Impl::push_packet(const void *data_, size_t size)
{
	auto *data = static_cast<const uint8_t *>(data_);

	const auto ensure_offset_table = [&]() -> uint32_t * {
		if (uma_offset_write_ptr)
			return uma_offset_write_ptr;
		if (uma_offset_direct && uma_slot_submitted && !uma_offset_synced_to_cpu && uma_dequant_offset[uma_submitted_slot])
		{
			auto *prev = static_cast<const uint32_t *>(device->map_host_buffer(
				*uma_dequant_offset[uma_submitted_slot], MEMORY_ACCESS_READ_BIT));
			if (prev)
			{
				memcpy(dequant_offset_buffer_cpu.data(), prev, size_t(block_count_32x32) * sizeof(uint32_t));
				device->unmap_host_buffer(*uma_dequant_offset[uma_submitted_slot], MEMORY_ACCESS_READ_BIT);
			}
			uma_offset_synced_to_cpu = true;
		}
		return dequant_offset_buffer_cpu.data();
	};

	uint32_t *offset_table = ensure_offset_table();

	while (size >= sizeof(BitstreamHeader))
	{
		auto *header = reinterpret_cast<const BitstreamHeader *>(data);

		if (header->extended != 0)
		{
			auto *seq = reinterpret_cast<const BitstreamSequenceHeader *>(header);

			if (sizeof(*header) > size)
			{
				LOGE("Parsing sequence header, but only %zu bytes left to parse.\n", size);
				return false;
			}

			if (seq->chroma_resolution != int(chroma))
			{
				LOGE("Chroma resolution mismatch!\n");
				return false;
			}

			uint8_t diff = (header->sequence - last_seq) & SequenceCountMask;
			if (last_seq != UINT32_MAX && diff > (SequenceCountMask / 2))
			{
				return true;
			}

			if (last_seq == UINT32_MAX || diff != 0)
			{
				clear();
				last_seq = header->sequence;
				offset_table = ensure_offset_table();
			}

			if (seq->code == BITSTREAM_EXTENDED_CODE_START_OF_FRAME)
			{
				if (seq->width_minus_1 + 1 != width || seq->height_minus_1 + 1 != height)
				{
					LOGE("Dimension mismatch in seq packet, (%d, %d) != (%d, %d)\n",
					     seq->width_minus_1 + 1, seq->height_minus_1 + 1, width, height);
					return false;
				}

				total_blocks_in_sequence = int(seq->total_blocks);
			}
			else
			{
				LOGE("Unrecognized sequence header mode %u.\n", seq->code);
				return false;
			}

			data += sizeof(*header);
			size -= sizeof(*header);

			continue;
		}

		const uint32_t pwords = header->payload_words;
		size_t packet_size = size_t(pwords) * sizeof(uint32_t);

		if (packet_size > size)
		{
			LOGE("Packet header states %zu bytes, but only %zu bytes left to parse.\n", packet_size, size);
			return false;
		}

		bool restart;

		if (last_seq == UINT32_MAX)
		{
			restart = true;
		}
		else
		{
			uint8_t diff = (header->sequence - last_seq) & SequenceCountMask;
			if (diff > (SequenceCountMask / 2))
			{
				return true;
			}
			restart = diff != 0;
		}

		if (restart)
		{
			clear();
			last_seq = header->sequence;
			offset_table = ensure_offset_table();
		}

		if (header->block_index >= uint32_t(block_count_32x32))
		{
			LOGE("block_index %u is out of bounds (>= %d).\n", header->block_index, block_count_32x32);
			return false;
		}

		if (sizeof(*header) / sizeof(uint32_t) > pwords)
		{
			LOGE("payload_words is not large enough.\n");
			return false;
		}

		auto &offset = offset_table[header->block_index];
		if (offset == UINT32_MAX)
		{
			const uint32_t block_offset = uint32_t(uma_payload_written_words + payload_data_cpu.size());
			decoded_blocks++;
			offset = block_offset;
			const uint8_t clb = block_to_comp_level_band[header->block_index];
			const uint8_t comp = (clb >> 2) & 3u;
			const uint8_t lvl = clb >> 4;
			const uint8_t bnd = clb & 3u;
			if (bnd != 0)
			{
				const uint32_t idx = active_hf_blocks[comp][lvl]++;
				if (packed_2x2_wavelets && lvl < DecompositionLevels - 1)
					active_block_list[comp][lvl][idx] = { block_dequant_coord[header->block_index], block_offset };
			}

			if (uma_payload_write_ptr &&
			    !uma_slot_submitted &&
			    payload_data_cpu.empty() &&
			    uma_payload_written_words + pwords + 4 <= uma_payload_cap_words)
			{
				memcpy(uma_payload_write_ptr + uma_payload_written_words, data, packet_size);
				uma_payload_written_words += pwords;
			}
			else
			{
				const auto *payload_words = reinterpret_cast<const uint32_t *>(data);
				payload_data_cpu.insert(payload_data_cpu.end(), payload_words, payload_words + pwords);
			}
		}

		data += packet_size;
		size -= packet_size;
	}

	if (size != 0)
	{
		LOGE("Did not consume packet completely.\n");
		return false;
	}

	return true;
}

void Decoder::Impl::init_block_meta()
{
	WaveletBuffers::init_block_meta();

	// Enable UMA zero-copy ring-buffered uploads only on SSBO paths (STORAGE_MODE 0 or 3)
	// so we never alias host-visible memory with linear 2D payload images.
	use_uma_upload = !use_readonly_texel_buffer && select_uma_buffer_domain(*device, uma_domain);
	uma_offset_direct = use_uma_upload && (uma_domain == BufferDomain::UMACachedCoherentPreferDevice);

	BufferCreateInfo info;
	info.size = block_count_32x32 * sizeof(uint32_t);
	if (use_uma_upload)
	{
		info.domain = uma_domain;
		info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
		for (uint32_t i = 0; i < kUmaRingSlots; i++)
		{
			uma_dequant_offset[i] = device->create_buffer(info);
			if (uma_dequant_offset[i])
				device->set_name(*uma_dequant_offset[i], "meta-buffer-uma");
		}
		if (packed_2x2_wavelets)
		{
			BufferCreateInfo ab_info;
			ab_info.size = std::max<VkDeviceSize>(64, VkDeviceSize(block_count_32x32) * 2 * sizeof(uint32_t));
			ab_info.domain = uma_domain;
			ab_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
			for (uint32_t i = 0; i < kUmaRingSlots; i++)
			{
				uma_active_blocks[i] = device->create_buffer(ab_info);
				if (uma_active_blocks[i])
					device->set_name(*uma_active_blocks[i], "active-blocks-uma");
			}
		}
		LOGI("Using UMA zero-copy ring-buffered uploads.\n");
	}
	else
	{
		info.domain = BufferDomain::Device;
		info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
		dequant_offset_buffer = device->create_buffer(info);
		device->set_name(*dequant_offset_buffer, "meta-buffer");
		if (packed_2x2_wavelets)
		{
			BufferCreateInfo ab_info;
			ab_info.size = std::max<VkDeviceSize>(64, VkDeviceSize(block_count_32x32) * 2 * sizeof(uint32_t));
			ab_info.domain = BufferDomain::Device;
			ab_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
			active_blocks_buffer = device->create_buffer(ab_info);
			if (active_blocks_buffer)
				device->set_name(*active_blocks_buffer, "active-blocks");
		}
	}

	const auto &feats = device->get_device_features();
	subgroup_ballot_mask =
		feats.vk11_props.subgroupSize >= 12 &&
		(!feats.vk13_features.subgroupSizeControl ||
		 feats.vk13_props.minSubgroupSize >= 12 ||
		 feats.driver_id == VK_DRIVER_ID_MOLTENVK ||
		 device->get_gpu_properties().vendorID == 0x106Bu);

	if (packed_2x2_wavelets)
	{
		build_static_edge_wg_lists();
		if (!edge_wg_list_cpu.empty())
		{
			BufferCreateInfo wg_info;
			wg_info.size = edge_wg_list_cpu.size() * sizeof(uint32_t);
			wg_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
			wg_info.domain = BufferDomain::Device;
			edge_wg_list_buffer = device->create_buffer(wg_info);
			if (edge_wg_list_buffer)
				device->set_name(*edge_wg_list_buffer, "idwt-edge-wg-list");
		}
	}

	dequant_offset_buffer_cpu.resize(block_count_32x32);
	block_to_comp_level_band.assign(block_count_32x32, 0);
	block_dequant_coord.assign(block_count_32x32, 0);
	for (int level = 0; level < DecompositionLevels; level++)
	{
		for (int component = 0; component < NumComponents; component++)
		{
			if (level == 0 && component != 0 && chroma == ChromaSubsampling::Chroma420)
				continue;
			int first_band = level == DecompositionLevels - 1 ? 0 : 1;
			size_t total_level_blocks = 0;
			for (int band = first_band; band < 4; band++)
			{
				const auto &m = block_meta[component][level][band];
				total_level_blocks += size_t(m.block_count_32x32);
				uint8_t clb = uint8_t((level << 4) | (component << 2) | band);
				const int stride = std::max(1, m.block_stride_32x32);
				for (int i = 0; i < m.block_count_32x32; i++)
				{
					block_to_comp_level_band[m.block_offset_32x32 + i] = clb;
					const uint32_t bx = uint32_t(i % stride);
					const uint32_t by = uint32_t(i / stride);
					block_dequant_coord[m.block_offset_32x32 + i] =
						(bx & 0xfffu) | ((by & 0xfffu) << 12) | (uint32_t(band) << 24);
				}
			}
			if (packed_2x2_wavelets && level < DecompositionLevels - 1)
				active_block_list[component][level].resize(total_level_blocks);
		}
	}

	payload_data_cpu.reserve(1024 * 1024);
}

bool Decoder::Impl::dequant(CommandBuffer &cmd)
{
	DequantizerPushData push = {};

	// Pascal benefits on sparse input, but this tradeoff did not help Tegra or RADV.
	const auto &gpu = device->get_gpu_properties();
	const bool direct_sign_prefix = gpu.vendorID == VENDOR_ID_NVIDIA &&
		gpu.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
	cmd.set_specialization_constant_mask(direct_sign_prefix ? 1 : 0);
	if (direct_sign_prefix)
		cmd.set_specialization_constant(0, true);
	cmd.enable_subgroup_size_control(true);

	if (device->supports_subgroup_size_log2(true, 4, 7))
	{
		cmd.set_subgroup_size_log2(true, 4, 7);
	}
	else if (device->supports_subgroup_size_log2(true, 2, 7))
	{
		cmd.set_subgroup_size_log2(true, 2, 7);
	}
	else
	{
		LOGE("No compatible subgroup size config.\n");
		return false;
	}

	const Buffer *active_dequant_offset = use_uma_upload ? uma_dequant_offset[uma_slot].get() : dequant_offset_buffer.get();
	const Buffer *active_payload_data = use_uma_upload ? uma_payload_data[uma_slot].get() : payload_data.get();
	if (!active_dequant_offset)
		return false;
	if (!payload_r8_image && !use_readonly_texel_buffer && !active_payload_data)
		return false;

	int storage_mode;
	if (payload_r8_image && payload_r16_image && payload_r32_image)
		storage_mode = 2;
	else if (use_readonly_texel_buffer)
		storage_mode = 1;
	else if (use_ssbo32_payload)
		storage_mode = 3;
	else
		storage_mode = 0;

	const bool debug_labels = device->get_device_features().supports_debug_utils;
	if (debug_labels)
		cmd.begin_region("DWT dequant");
	auto start_dequant = cmd.write_timestamp(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

	cmd.image_barrier(*wavelet_img_high_res, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
	                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
	                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
	                  VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

	if (wavelet_img_low_res)
	{
		cmd.image_barrier(*wavelet_img_low_res, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
		                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
		                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		                  VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
	}

	if (wavelet_img_coarsest)
	{
		cmd.image_barrier(*wavelet_img_coarsest, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
		                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
		                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		                  VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
	}

	// De-quantize
	for (int level = 0; level < DecompositionLevels; level++)
	{
		const bool is_packed = level_is_packed_2x2(level);
		cmd.set_program(shaders.wavelet_dequant[storage_mode][is_packed ? 1 : 0]);
		for (int component = 0; component < NumComponents; component++)
		{
			// Ignore top-level CbCr when doing 420 subsampling.
			if (level == 0 && component != 0 && chroma == ChromaSubsampling::Chroma420)
				continue;

			// Packed levels never zero empty blocks in dequant (iDWT skips empty blocks via payload_offsets);
			// levels with zero active HF blocks need no dequant dispatch at all.
			const uint32_t act_hf = active_hf_blocks[component][level];
			if (is_packed && act_hf == 0)
				continue;

			if (debug_labels)
			{
				char label[128];
				snprintf(label, sizeof(label), "level %d - component %d", level, component);
				cmd.begin_region(label);
			}

			int first_band = level == DecompositionLevels - 1 ? 0 : 1;
			push.resolution.x = int32_t(get_level_width(level));
			push.resolution.y = int32_t(get_level_height(level));
			push.output_layer = first_band;
			push.block_offset_32x32 = is_packed ? int32_t(active_block_offset[component][level])
			                                    : block_meta[component][level][first_band].block_offset_32x32;
			push.block_stride_32x32 = block_meta[component][level][first_band].block_stride_32x32;
			push.band_stride_32x32 = block_meta[component][level][first_band].block_count_32x32;
			cmd.push_constants(&push, 0, sizeof(push));

			cmd.set_storage_texture(0, 0, *component_layer_views[component][level]);
			if (is_packed)
			{
				const Buffer *active_blocks_buf = use_uma_upload ? uma_active_blocks[uma_slot].get() : active_blocks_buffer.get();
				if (!active_blocks_buf)
					return false;
				cmd.set_storage_buffer(0, 1, *active_blocks_buf);
			}
			else
			{
				cmd.set_storage_buffer(0, 1, *active_dequant_offset);
			}

			if (payload_r8_image && payload_r16_image && payload_r32_image)
			{
				cmd.set_texture(0, 2, payload_r32_image->get_view());
				cmd.set_texture(0, 3, payload_r16_image->get_view());
				cmd.set_texture(0, 4, payload_r8_image->get_view());
			}
			else if (use_readonly_texel_buffer)
			{
				cmd.set_buffer_view(0, 2, *payload_u32_view);
				cmd.set_buffer_view(0, 3, *payload_u16_view);
				cmd.set_buffer_view(0, 4, *payload_u8_view);
			}
			else
				cmd.set_storage_buffer(0, 2, *active_payload_data);

			if (is_packed)
				cmd.dispatch(act_hf, 1, 1);
			else
				cmd.dispatch((push.resolution.x + 31) / 32, (push.resolution.y + 31) / 32, 4 - first_band);

			if (debug_labels)
				cmd.end_region();
		}
	}

	cmd.barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
	            fragment_path ? VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT : VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
	            VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);

	auto end_dequant = cmd.write_timestamp(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
	if (debug_labels)
		cmd.end_region();
	cmd.set_specialization_constant_mask(0);
	cmd.enable_subgroup_size_control(false);
	device->register_time_interval("GPU", std::move(start_dequant), std::move(end_dequant), "Dequant");

	return true;
}

bool Decoder::Impl::idwt_fragment(CommandBuffer &cmd, const ViewBuffers &views)
{
	const auto add_discard = [&](const Vulkan::Image *img) {
		if (img)
		{
			cmd.image_barrier(*img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
			                  VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
			                  VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);
		}
	};

	const auto add_read_only = [&](const Vulkan::Image *img) {
		if (img)
		{
			cmd.image_barrier(*img, VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL,
			                  VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
			                  VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
		}
	};

	cmd.begin_barrier_batch();
	for (auto &level : fragment.levels)
	{
		for (auto &vert : level.vert)
			for (auto &comp : vert)
				add_discard(comp.get());
		for (auto &comp : level.horiz)
			add_discard(comp.get());
	}
	cmd.end_barrier_batch();

	auto start_idwt = cmd.write_timestamp(VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);

	struct Push
	{
		float u_offset;
		float v_offset;
		float half_texel_offset_u;
		float half_texel_offset_v;
		float vp_scale;
		uint32_t pivot_size;
	} push = {};

	for (int input_level = DecompositionLevels - 1; input_level >= 0; input_level--)
	{
		int output_level = input_level - 1;

		char label[128];
		if (output_level >= 0)
			snprintf(label, sizeof(label), "Fragment iDWT level %u", output_level);
		else
			snprintf(label, sizeof(label), "Fragment iDWT final");
		cmd.begin_region(label);

		Vulkan::RenderPassInfo rp_info = {};

		bool has_chroma_output = output_level >= 0 || chroma == ChromaSubsampling::Chroma444;

		Vulkan::Program *vert_prog;
		Vulkan::Program *horiz_prog;

		if (has_chroma_output)
		{
			rp_info.store_attachments = 0x3;
			rp_info.num_color_attachments = 2;
			vert_prog = device->request_program(shaders.idwt_vs, shaders.idwt_fs[1]);
			horiz_prog = device->request_program(shaders.idwt_vs, shaders.idwt_fs[2]);
		}
		else
		{
			rp_info.store_attachments = 0x1;
			rp_info.num_color_attachments = 1;
			vert_prog = device->request_program(shaders.idwt_vs, shaders.idwt_fs[0]);
			horiz_prog = vert_prog;
		}

		// Vertical passes.
		for (int vert_pass = 0; vert_pass < 2; vert_pass++)
		{
			rp_info.color_attachments[0] = &fragment.levels[input_level].vert[vert_pass][0]->get_view();
			if (has_chroma_output)
				rp_info.color_attachments[1] = &fragment.levels[input_level].vert[vert_pass][1]->get_view();

			cmd.begin_render_pass(rp_info);
			cmd.set_program(vert_prog);
			cmd.set_opaque_sprite_state();
			cmd.set_specialization_constant_mask(0x1 | 0x8);
			cmd.set_specialization_constant(0, true);

			cmd.set_texture(0, 0, *fragment.levels[input_level].decoded[0][vert_pass + 0]);
			cmd.set_texture(0, 1, *fragment.levels[input_level].decoded[0][vert_pass + 2]);
			cmd.set_sampler(0, 2, *mirror_repeat_sampler);

			if (has_chroma_output)
			{
				cmd.set_texture(0, 3, *fragment.levels[input_level].decoded[1][vert_pass + 0]);
				cmd.set_texture(0, 4, *fragment.levels[input_level].decoded[1][vert_pass + 2]);
				cmd.set_texture(0, 5, *fragment.levels[input_level].decoded[2][vert_pass + 0]);
				cmd.set_texture(0, 6, *fragment.levels[input_level].decoded[2][vert_pass + 2]);
			}

			uint32_t render_width = rp_info.color_attachments[0]->get_view_width();
			uint32_t render_height = rp_info.color_attachments[0]->get_view_height();

			// Set mirror point.
			// Work around broken Mali r38.1 compiler.
			// If it sees negative texture offsets it breaks the output for whatever reason (!?!?!?!).
			auto *input_view = fragment.levels[input_level].decoded[0][0].get();
			push.u_offset = 0.0f;
			push.v_offset = -2.0f / float(input_view->get_view_height());
			push.half_texel_offset_u = 0.5f / float(input_view->get_view_width());
			push.half_texel_offset_v = 0.5f / float(input_view->get_view_height());
			push.vp_scale = cmd.get_viewport().height;
			push.pivot_size = render_height;
			cmd.push_constants(&push, 0, sizeof(push));

			// Render top edge condition.
			cmd.set_specialization_constant(3, -1);
			cmd.set_scissor({{ 0, 0 }, { render_width, 8 }});
			cmd.draw(3);

			// Render normal path
			cmd.set_specialization_constant(3, 0);
			cmd.set_scissor({{ 0, 8 }, { render_width, render_height - 16 }});
			cmd.draw(3);

			// Render bottom edge condition
			cmd.set_specialization_constant(3, +1);
			cmd.set_scissor({{ 0, int(render_height) - 8 }, { render_width, 8 }});
			cmd.draw(3);

			cmd.end_render_pass();
		}

		cmd.begin_barrier_batch();
		for (auto &vert : fragment.levels[input_level].vert)
			for (auto &comp : vert)
				add_read_only(comp.get());
		cmd.end_barrier_batch();

		if (has_chroma_output)
		{
			rp_info.num_color_attachments = 3;
			rp_info.store_attachments = 0x7;
		}
		else
		{
			rp_info.num_color_attachments = 1;
			rp_info.store_attachments = 0x1;
		}

		for (uint32_t comp = 0; comp < rp_info.num_color_attachments; comp++)
		{
			if (output_level < 0 || (output_level == 0 && chroma == ChromaSubsampling::Chroma420 && comp != 0))
				rp_info.color_attachments[comp] = views.planes[comp];
			else
				rp_info.color_attachments[comp] = &fragment.levels[output_level].horiz[comp]->get_view();
		}

		cmd.begin_render_pass(rp_info);
		cmd.set_program(horiz_prog);
		cmd.set_opaque_sprite_state();
		cmd.set_specialization_constant_mask(0xf);
		cmd.set_specialization_constant(0, false);
		cmd.set_specialization_constant(1, output_level < 0);
		cmd.set_specialization_constant(2, output_level < 0 || (output_level == 0 && chroma == ChromaSubsampling::Chroma420));

		cmd.set_texture(0, 0, fragment.levels[input_level].vert[0][0]->get_view());
		cmd.set_texture(0, 1, fragment.levels[input_level].vert[1][0]->get_view());
		cmd.set_sampler(0, 2, *mirror_repeat_sampler);

		if (has_chroma_output)
		{
			cmd.set_texture(0, 3, fragment.levels[input_level].vert[0][1]->get_view());
			cmd.set_texture(0, 4, fragment.levels[input_level].vert[1][1]->get_view());
		}

		uint32_t aligned_render_width = aligned_width >> (output_level + 1);
		uint32_t aligned_render_height = aligned_height >> (output_level + 1);

		// Chroma output might be smaller than Y in output_level == 0 due to not using alignment.
		// This is reflected in the actual render area, which is equal to default viewport.
		auto render_width = uint32_t(cmd.get_viewport().width);
		auto render_height = uint32_t(cmd.get_viewport().height);

		// In case we're rendering to an output texture,
		// the render area might be smaller than we expect for purposes of alignment.
		// Use properly scaled viewport that we scissor away as needed.
		cmd.set_viewport({ 0, 0, float(aligned_render_width), float(aligned_render_height), 0, 1 });

		// Set mirror point.
		auto *input_view = &fragment.levels[input_level].vert[0][0]->get_view();
		push.u_offset = -2.0f / float(input_view->get_view_width());
		push.v_offset = 0.0f;
		push.half_texel_offset_u = 0.5f / float(input_view->get_view_width());
		push.half_texel_offset_v = 0.5f / float(input_view->get_view_height());
		push.vp_scale = cmd.get_viewport().width;
		push.pivot_size = aligned_render_width;
		cmd.push_constants(&push, 0, sizeof(push));

		// Render left edge condition.
		cmd.set_specialization_constant(3, -1);
		cmd.set_scissor({{ 0, 0 }, { 8, render_height }});
		cmd.draw(3);

		// Render normal condition
		cmd.set_specialization_constant(3, 0);
		cmd.set_scissor({{ 8, 0 }, { std::min<uint32_t>(render_width - 8, aligned_render_width - 16), render_height }});
		cmd.draw(3);

		uint32_t aligned_x = aligned_render_width - 8;
		if (aligned_x < render_width)
		{
			// Render right edge condition
			cmd.set_specialization_constant(3, +1);
			cmd.set_scissor({{ int(aligned_x), 0 }, { render_width - aligned_x, render_height }});
			cmd.draw(3);
		}

		cmd.end_render_pass();

		// If chroma is subsampled, we cannot render the fully padded region in one render pass due to
		// rules regarding renderArea. renderArea cannot exceed the smallest image in the render pass.
		// We cannot use subpasses either, so split the render pass, but that's mostly fine,
		// since renderArea is non-overlapping.
		if (output_level == 0 && chroma == ChromaSubsampling::Chroma420)
		{
			rp_info.num_color_attachments = 1;
			rp_info.store_attachments = 0x1;

			VkMemoryBarrier2 by_region = { VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
			by_region.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
			by_region.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
			by_region.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
			by_region.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
			VkDependencyInfo dep = { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
			dep.memoryBarrierCount = 1;
			dep.pMemoryBarriers = &by_region;
			dep.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

			// Need vertical fixup (very common for 1080p).
			if (rp_info.color_attachments[1]->get_view_height() < rp_info.color_attachments[0]->get_view_height())
			{
				// Insert a simple by_region barrier to ensure we follow Vulkan rules for RW access.
				cmd.barrier(dep);

				rp_info.render_area.extent.width = rp_info.color_attachments[0]->get_view_width();
				rp_info.render_area.extent.height =
						rp_info.color_attachments[0]->get_view_height() -
						rp_info.color_attachments[1]->get_view_height();
				rp_info.render_area.offset.x = 0;
				rp_info.render_area.offset.y = rp_info.color_attachments[1]->get_view_height();

				cmd.begin_render_pass(rp_info);
				cmd.set_program(device->request_program(shaders.idwt_vs, shaders.idwt_fs[0]));
				cmd.set_opaque_sprite_state();
				cmd.set_texture(0, 0, fragment.levels[input_level].vert[0][0]->get_view());
				cmd.set_texture(0, 1, fragment.levels[input_level].vert[1][0]->get_view());
				cmd.set_sampler(0, 2, *mirror_repeat_sampler);
				cmd.set_viewport({ 0, 0, float(aligned_render_width), float(aligned_render_height), 0, 1 });
				cmd.set_specialization_constant_mask(0x8);
				cmd.push_constants(&push, 0, sizeof(push));
				cmd.set_specialization_constant(3, 1); // Always consider edge handling.
				cmd.draw(3);
				cmd.end_render_pass();
			}

			// Need horizontal fixup (very rare).
			if (rp_info.color_attachments[1]->get_view_width() < rp_info.color_attachments[0]->get_view_width())
			{
				cmd.barrier(dep);

				rp_info.render_area.extent.width =
						rp_info.color_attachments[0]->get_view_width() -
						rp_info.color_attachments[1]->get_view_width();
				rp_info.render_area.extent.height = rp_info.color_attachments[0]->get_view_height();
				rp_info.render_area.offset.x = rp_info.color_attachments[1]->get_view_width();
				rp_info.render_area.offset.y = 0;

				cmd.begin_render_pass(rp_info);
				cmd.set_program(device->request_program(shaders.idwt_vs, shaders.idwt_fs[0]));
				cmd.set_opaque_sprite_state();
				cmd.set_texture(0, 0, fragment.levels[input_level].vert[0][0]->get_view());
				cmd.set_texture(0, 1, fragment.levels[input_level].vert[1][0]->get_view());
				cmd.set_sampler(0, 2, *mirror_repeat_sampler);
				cmd.set_viewport({ 0, 0, float(aligned_render_width), float(aligned_render_height), 0, 1 });
				cmd.set_specialization_constant_mask(0x8);
				cmd.push_constants(&push, 0, sizeof(push));
				cmd.set_specialization_constant(3, 1); // Always consider edge handling.
				cmd.draw(3);
				cmd.end_render_pass();
			}
		}

		if (output_level >= 0)
		{
			cmd.begin_barrier_batch();
			for (auto &comp: fragment.levels[output_level].horiz)
				add_read_only(comp.get());
			cmd.end_barrier_batch();
		}

		cmd.end_region();
	}

	auto end_idwt = cmd.write_timestamp(VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
	device->register_time_interval("GPU", std::move(start_idwt), std::move(end_idwt), "iDWT fragment");

	cmd.set_specialization_constant_mask(0);

	// Avoid WAR hazard for dequantization.
	cmd.barrier(VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0);

	return true;
}

bool Decoder::Impl::idwt(CommandBuffer &cmd, const ViewBuffers &views)
{
	const int prec = Configuration::get().get_precision();
	const Buffer *active_dequant_offset = use_uma_upload ? uma_dequant_offset[uma_slot].get() : dequant_offset_buffer.get();
	if (packed_2x2_wavelets && !active_dequant_offset)
		return false;

	cmd.enable_subgroup_size_control(false);

	auto start_idwt = cmd.write_timestamp(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

	struct
	{
		ivec2 resolution;
		vec2 inv_resolution;
		int32_t block_offset_32x32;
		int32_t block_stride_32x32;
		int32_t band_stride_32x32;
		int32_t wg_offset_x;
		int32_t wg_offset_y;
	} push = {};

	// Prewarm the small fixed set of packed iDWT pipelines once so scene changes never trigger
	// mid-stream shader compilation stalls on the decode thread.
	if (packed_2x2_wavelets)
	{
		const auto prewarm_one = [&](Program *prog, bool final_out, float rscale,
		                             bool edge_pass, uint32_t band_mask, bool indirect_list, bool mask_check) {
			if (!prog)
				return;
			cmd.set_program(prog);
			const uint32_t smask = (final_out ? 0x3u : 0x0u) | 0x7cu;
			cmd.set_specialization_constant_mask(smask);
			if (final_out)
			{
				cmd.set_specialization_constant(0, true);
				cmd.set_specialization_constant(1, rscale);
			}
			cmd.set_specialization_constant(2, edge_pass);
			cmd.set_specialization_constant(3, band_mask);
			cmd.set_specialization_constant(4, indirect_list);
			cmd.set_specialization_constant(5, mask_check);
			cmd.set_specialization_constant(6, subgroup_ballot_mask);
			DeferredPipelineCompile compile = {};
			cmd.extract_pipeline_state(compile);
			if (!prog->get_pipeline(compile.hash).pipeline)
				CommandBuffer::build_compute_pipeline(device, compile, CommandBuffer::CompileMode::Sync);
		};

		const auto prewarm_level = [&](bool final_out, bool split) {
			Program *p = shaders.idwt[prec][1][final_out ? 0 : 1];
			const float scale = final_out ? views.range_scale : 1.0f;
			for (int variant = 0; variant < 3; variant++)
			{
				const uint32_t bands = variant == 0 ? 0u : 7u;
				const bool mask_check = variant == 2;
				if (split)
					prewarm_one(p, final_out, scale, false, bands, false, mask_check);
				prewarm_one(p, final_out, scale, true, bands, split, mask_check);
			}
		};

		if (!prewarmed_idwt_intermediate)
		{
			bool split = false, direct = false;
			for (int level = 1; level < DecompositionLevels - 1; level++)
			{
				split |= idwt_level_layouts[level].use_split;
				direct |= !idwt_level_layouts[level].use_split;
			}
			if (split)
				prewarm_level(false, true);
			if (direct)
				prewarm_level(false, false);
			prewarmed_idwt_intermediate = true;
		}
		if (!prewarmed_idwt_final || prewarmed_range_scale != views.range_scale)
		{
			const bool luma_split = idwt_level_layouts[0].use_split;
			prewarm_level(true, luma_split);
			if (chroma == ChromaSubsampling::Chroma420 && idwt_level_layouts[1].use_split != luma_split)
				prewarm_level(true, !luma_split);
			prewarmed_idwt_final = true;
			prewarmed_range_scale = views.range_scale;
		}
		cmd.set_specialization_constant_mask(0);
	}

	for (int input_level = DecompositionLevels - 1; input_level >= 0; input_level--)
	{
		const bool packed_in = level_is_packed_2x2(input_level);
		// Transposed.
		push.resolution.x = int32_t(get_level_height(input_level));
		push.resolution.y = int32_t(get_level_width(input_level));
		push.inv_resolution.x = 1.0f / float(push.resolution.x);
		push.inv_resolution.y = 1.0f / float(push.resolution.y);

		const int nx = (push.resolution.x + 15) / 16;
		const int ny = (push.resolution.y + 15) / 16;

		const auto dispatch_component = [&](int c, bool final_output, const ImageView &dst_view) {
			const bool packed_out = !final_output && level_is_packed_2x2(input_level - 1);
			cmd.set_program(shaders.idwt[prec][packed_in ? 1 : 0][packed_out ? 1 : 0]);
			cmd.set_storage_texture(0, 1, dst_view);

			if (packed_in)
			{
				const auto &layout = idwt_level_layouts[input_level];
				const auto &meta = block_meta[c][input_level][1];
				push.block_offset_32x32 = meta.block_offset_32x32;
				push.block_stride_32x32 = meta.block_stride_32x32;
				push.band_stride_32x32 = meta.block_count_32x32;
				cmd.set_texture(0, 0, *component_layer_views[c][input_level]);
				cmd.set_storage_buffer(0, 2, *active_dequant_offset);
				cmd.set_storage_buffer(0, 3, edge_wg_list_buffer ? *edge_wg_list_buffer : *active_dequant_offset);

				uint32_t spec_mask = (final_output ? 0x3u : 0x0u) | 0x7cu;
				cmd.set_specialization_constant_mask(spec_mask);
				if (final_output)
				{
					cmd.set_specialization_constant(0, true);
					cmd.set_specialization_constant(1, views.range_scale);
				}
				const uint32_t act_hf = active_hf_blocks[c][input_level];
				const uint32_t total_hf = 3u * uint32_t(meta.block_count_32x32);
				const bool ll_only = act_hf == 0;
				const bool mask_check = act_hf > 0 && act_hf < total_hf;
				cmd.set_specialization_constant(3, ll_only ? 0u : 7u);
				cmd.set_specialization_constant(5, mask_check);
				cmd.set_specialization_constant(6, subgroup_ballot_mask);

				if (layout.use_split)
				{
					cmd.set_specialization_constant(2, false);
					cmd.set_specialization_constant(4, false);
					push.wg_offset_x = 1;
					push.wg_offset_y = 1;
					cmd.push_constants(&push, 0, sizeof(push));
					cmd.dispatch(uint32_t(layout.ix_end - 1), uint32_t(layout.iy_end - 1), 1);

					if (layout.edge_count > 0)
					{
						cmd.set_specialization_constant(2, true);
						cmd.set_specialization_constant(4, true);
						push.wg_offset_x = int32_t(layout.edge_offset);
						push.wg_offset_y = 0;
						cmd.push_constants(&push, 0, sizeof(push));
						cmd.dispatch(layout.edge_count, 1, 1);
					}
				}
				else
				{
					cmd.set_specialization_constant(2, true);
					cmd.set_specialization_constant(4, false);
					push.wg_offset_x = 0;
					push.wg_offset_y = 0;
					cmd.push_constants(&push, 0, sizeof(push));
					cmd.dispatch(nx, ny, 1);
				}
			}
			else
			{
				cmd.set_specialization_constant_mask(final_output ? 3 : 0);
				if (final_output)
				{
					cmd.set_specialization_constant(0, true);
					cmd.set_specialization_constant(1, views.range_scale);
				}
				cmd.push_constants(&push, 0, 16);
				cmd.set_texture(0, 0, *component_layer_views[c][input_level], *mirror_repeat_sampler);
				cmd.dispatch(nx, ny, 1);
			}
		};

		const bool debug_labels = device->get_device_features().supports_debug_utils;
		if (input_level == 0)
		{
			if (chroma == ChromaSubsampling::Chroma444)
			{
				for (int c = 0; c < NumComponents; c++)
				{
					if (debug_labels)
					{
						char label[64];
						snprintf(label, sizeof(label), "iDWT final, component %u", c);
						cmd.begin_region(label);
					}
					dispatch_component(c, true, *views.planes[c]);
					if (debug_labels)
						cmd.end_region();
				}
			}
			else
			{
				if (debug_labels)
					cmd.begin_region("iDWT final");
				dispatch_component(0, true, *views.planes[0]);
				if (debug_labels)
					cmd.end_region();
			}
		}
		else
		{
			for (int c = 0; c < NumComponents; c++)
			{
				const bool is_final_chroma = chroma == ChromaSubsampling::Chroma420 && c != 0 && input_level == 1;
				const ImageView &dst = is_final_chroma ? *views.planes[c] : *component_ll_views[c][input_level - 1];

				if (debug_labels)
				{
					char label[64];
					snprintf(label, sizeof(label), "iDWT level %u, component %u", input_level - 1, c);
					cmd.begin_region(label);
				}
				dispatch_component(c, is_final_chroma, dst);
				if (debug_labels)
					cmd.end_region();
			}
		}

		cmd.set_specialization_constant_mask(0);
		// The next level samples this level's output. The final level writes the planes
		// consumed by Moonlight, which synchronizes them when it submits the render.
		if (input_level != 0)
			cmd.barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
			            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
	}

	auto end_idwt = cmd.write_timestamp(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
	device->register_time_interval("GPU", std::move(start_idwt), std::move(end_idwt), "iDWT");
	return true;
}

bool Decoder::Impl::has_pristine_bands(int bands, const uint32_t *active_block_mask, size_t word_count) const
{
	// Account for 4:2:0 where level0 will not have packets.
	// It's somewhat meaningless to ask for pristine bands all the way up to that point though.
	assert(bands < DecompositionLevels);

	// This analysis assumes that there are no "null" blocks present.
	// For the lowest frequency bands, that is vanishingly unlikely to happen,
	// and worst case we get a false positive rejection.
	// TODO: It's possible to improve this by sending the "active block mask"
	// as sideband data from encoder if need be.

	const uint32_t *offset_table = uma_offset_write_ptr ? uma_offset_write_ptr : dequant_offset_buffer_cpu.data();
	const auto block_is_missing = [&](uint32_t block_index)
	{
		if (offset_table[block_index] != UINT32_MAX)
			return false;

		uint32_t word_index = block_index / 32;

		if (!active_block_mask || word_index >= word_count)
			return true;

		// If the block wasn't expected to be active anyway, just pass it through.
		return ((active_block_mask[word_index] >> (block_index % 32)) & 1) != 0;
	};

	for (int band = 0; band < bands; band++)
	{
		for (auto &component : block_meta)
		{
			if (band == 0)
			{
				auto &meta = component[DecompositionLevels - 1][0];
				for (int i = 0; i < meta.block_count_32x32; i++)
					if (block_is_missing(meta.block_offset_32x32 + i))
						return false;
			}
			else
			{
				// If we can reconstruct the LH, HL, HH bands, we can generate the higher-resolution LL band.
				for (int high_freq_bands = 1; high_freq_bands < 4; high_freq_bands++)
				{
					auto &meta = component[DecompositionLevels - band][high_freq_bands];
					for (int i = 0; i < meta.block_count_32x32; i++)
						if (block_is_missing(meta.block_offset_32x32 + i))
							return false;
				}
			}
		}
	}

	return true;
}

bool Decoder::Impl::decode_is_ready(bool allow_partial_frame, int pristine_bands, float received_ratio,
                                    const uint32_t *active_block_mask, size_t word_count) const
{
	if (decoded_frame_for_current_sequence)
		return false;

	if (last_seq == UINT32_MAX)
		return false;

	if (decoded_blocks < total_blocks_in_sequence)
	{
		if (!allow_partial_frame)
			return false;

		if (!has_pristine_bands(pristine_bands, active_block_mask, word_count))
			return false;
		if (float(decoded_blocks) <= float(total_blocks_in_sequence) * received_ratio)
			return false;
	}

	return true;
}

bool Decoder::Impl::decode_is_ready(bool allow_partial_frame) const
{
	// At the very least, we want some LL bands to be received properly,
	// otherwise we get extreme artifacts.
	return decode_is_ready(allow_partial_frame, 2, 0.9f, nullptr, 0);
}

bool Decoder::Impl::decode(CommandBuffer &cmd, const ViewBuffers &views)
{
	if (packed_2x2_wavelets && edge_wg_list_buffer && !edge_wg_list_uploaded)
	{
		VkDeviceSize wg_size = edge_wg_list_cpu.size() * sizeof(uint32_t);
		memcpy(cmd.update_buffer(*edge_wg_list_buffer, 0, wg_size), edge_wg_list_cpu.data(), wg_size);
		cmd.barrier(VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
		            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
		edge_wg_list_uploaded = true;
	}

	if (use_uma_upload)
	{
		if (!upload_payload_uma(cmd))
			return false;
	}
	else
	{
		cmd.begin_region("Decode uploads");
		{
			upload_payload(cmd);

			memcpy(cmd.update_buffer(*dequant_offset_buffer, 0,
			                         dequant_offset_buffer_cpu.size() * sizeof(dequant_offset_buffer_cpu.front())),
			       dequant_offset_buffer_cpu.data(), dequant_offset_buffer_cpu.size() * sizeof(dequant_offset_buffer_cpu.front()));

			if (packed_2x2_wavelets && active_blocks_buffer)
			{
				uint32_t total_entries = 0;
				for (int l = 0; l < DecompositionLevels - 1; l++)
					for (int c = 0; c < NumComponents; c++)
						total_entries += active_hf_blocks[c][l];

				if (total_entries > 0)
				{
					auto *dst = static_cast<ActiveBlockEntry *>(cmd.update_buffer(*active_blocks_buffer, 0,
					                                                              VkDeviceSize(total_entries) * sizeof(ActiveBlockEntry)));
					uint32_t entry_offset = 0;
					for (int l = 0; l < DecompositionLevels - 1; l++)
					{
						for (int c = 0; c < NumComponents; c++)
						{
							active_block_offset[c][l] = entry_offset;
							const uint32_t count = active_hf_blocks[c][l];
							if (count > 0)
							{
								memcpy(dst + entry_offset, active_block_list[c][l].data(), count * sizeof(ActiveBlockEntry));
								entry_offset += count;
							}
						}
					}
				}
			}

			cmd.barrier(VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
			            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
			            VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
			            (use_readonly_texel_buffer ? VK_ACCESS_2_SHADER_SAMPLED_READ_BIT : 0));
		}
		cmd.end_region();
	}

	if (!dequant(cmd))
		return false;

	if (fragment_path)
	{
		if (!use_uma_upload)
			cmd.barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, 0, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);

		if (!idwt_fragment(cmd, views))
			return false;
	}
	else
	{
		if (!idwt(cmd, views))
			return false;

		if (!use_uma_upload)
			cmd.barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, 0, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
	}

	decoded_frame_for_current_sequence = true;
	return true;
}

void Decoder::Impl::clear()
{
	begin_uma_payload_write();
	uma_offset_synced_to_cpu = false;
	if (uma_offset_write_ptr)
		memset(uma_offset_write_ptr, 0xff, size_t(block_count_32x32) * sizeof(uint32_t));
	else
		memset(dequant_offset_buffer_cpu.data(), 0xff, dequant_offset_buffer_cpu.size() * sizeof(uint32_t));
	memset(active_hf_blocks, 0, sizeof(active_hf_blocks));
	memset(active_block_offset, 0, sizeof(active_block_offset));
	decoded_blocks = 0;
	last_seq = UINT32_MAX;
	decoded_frame_for_current_sequence = false;
	total_blocks_in_sequence = block_count_32x32;
	payload_data_cpu.clear();
}

bool Decoder::device_prefers_fragment_path(Vulkan::Device &device)
{
	switch (device.get_device_features().driver_id)
	{
	// QCOM hardware struggles with compute in general and prefers fragment.
	// Turnip seems to like compute path just fine though ...
	case VK_DRIVER_ID_QUALCOMM_PROPRIETARY:
		return true;

	// Mali heavily favors texture sampling over LS heavy content.
	case VK_DRIVER_ID_ARM_PROPRIETARY:
	case VK_DRIVER_ID_MESA_PANVK:
		return true;

	default:
		return false;
	}
}

void Decoder::Impl::check_linear_texture_support()
{
	if (!use_readonly_texel_buffer)
		return;

	// Texel buffers hit LS path on at least Mali, and most likely they hit slow paths on most mobile IHVs.
	// Try to promote to linear 2D images instead if we can get away with it.
	// Texture sampling performance is what mobile IHVs tend to optimize for.
	const struct
	{
		VkFormat fmt;
		uint32_t width;
		ImageHandle *out_handle;
	} reqs[] = {
		{ VK_FORMAT_R8_UINT, 4096, &payload_r8_image },
		{ VK_FORMAT_R16_UINT, 2048, &payload_r16_image },
		{ VK_FORMAT_R32_UINT, 1024, &payload_r32_image }
	};

	// Just assume this works. Can't imagine any GPU where this wouldn't work tightly packed.

	BufferCreateInfo bufinfo;
	bufinfo.size = 4 * 1024 * 1024;
	bufinfo.usage =
			VK_BUFFER_USAGE_TRANSFER_DST_BIT |
			VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
			VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT;
	bufinfo.domain = BufferDomain::Device;
	bufinfo.allocation_requirements.alignment = 64 * 1024;
	bufinfo.allocation_requirements.size = 4 * 1024 * 1024;
	bufinfo.allocation_requirements.memoryTypeBits = UINT32_MAX;
	payload_data = device->create_buffer(bufinfo);
	device->set_name(*payload_data, "payload-data");

	const auto *alias = &payload_data->get_allocation();

	// Try to force all linear images to alias each other.
	for (auto &req : reqs)
	{
		VkImageFormatProperties2 props2 = { VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2 };
		if (device->get_image_format_properties(req.fmt, VK_IMAGE_TYPE_2D, VK_IMAGE_TILING_LINEAR,
		                                        VK_IMAGE_USAGE_SAMPLED_BIT, 0,
		                                        nullptr, &props2))
		{
			if (props2.imageFormatProperties.maxExtent.width >= req.width &&
			    props2.imageFormatProperties.maxExtent.height >= 1024)
			{
				auto info = ImageCreateInfo::immutable_2d_image(req.width, 1024, req.fmt);
				info.domain = ImageDomain::LinearHost;
				info.usage = VK_IMAGE_USAGE_SAMPLED_BIT;
				info.num_memory_aliases = 1;
				info.layout = ImageLayout::General;
				info.memory_aliases = &alias;
				*req.out_handle = device->create_image(info);
			}
		}
	}

	if (payload_r8_image && payload_r16_image && payload_r32_image)
		LOGI("Using linear textures instead of texel buffers.\n");
}

bool Decoder::init(Vulkan::Device *device, int width, int height, ChromaSubsampling chroma_, bool fragment_path_)
{
	auto ops = device->get_device_features().vk11_props.subgroupSupportedOperations;
	constexpr VkSubgroupFeatureFlags required_features =
			VK_SUBGROUP_FEATURE_VOTE_BIT |
			VK_SUBGROUP_FEATURE_BALLOT_BIT |
			VK_SUBGROUP_FEATURE_ARITHMETIC_BIT |
			VK_SUBGROUP_FEATURE_SHUFFLE_BIT |
			VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT |
			VK_SUBGROUP_FEATURE_BASIC_BIT;

	if ((ops & required_features) != required_features)
	{
		LOGE("There are missing subgroup features. Device supports #%x, but requires #%x.\n",
		     ops, required_features);
		return false;
	}

	// The decoder is more lenient.
	if (!device->supports_subgroup_size_log2(true, 2, 7))
	{
		LOGE("Device doesn't support basic subgroup size control.\n");
		return false;
	}

	if (!impl->init(device, width, height, chroma_, fragment_path_, !fragment_path_))
	{
		LOGE("Failed to initialize.\n");
		return false;
	}

	if (!device->get_device_features().vk12_features.storageBuffer8BitAccess &&
	    !impl->use_readonly_texel_buffer &&
	    !impl->use_ssbo32_payload)
	{
		LOGE("Device doesn't support 8-bit storage or large texel buffers.\n");
		return false;
	}

	impl->check_linear_texture_support();

	clear();
	return true;
}

void Decoder::clear()
{
	impl->clear();
}

bool Decoder::push_packet(const void *data, size_t size)
{
	return impl->push_packet(data, size);
}

bool Decoder::decode(Vulkan::CommandBuffer &cmd, const ViewBuffers &views)
{
	return impl->decode(cmd, views);
}

bool Decoder::decode_is_ready(bool allow_partial_frame) const
{
	return impl->decode_is_ready(allow_partial_frame);
}

bool Decoder::decode_is_ready(bool allow_partial_frame, int num_pristine_bands, float minimum_packet_ratio,
                              const uint32_t *active_block_mask, size_t word_count) const
{
	return impl->decode_is_ready(allow_partial_frame, num_pristine_bands, minimum_packet_ratio, active_block_mask, word_count);
}
}
