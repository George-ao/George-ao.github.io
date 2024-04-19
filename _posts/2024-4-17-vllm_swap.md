---
layout: post
title: Swap policy in vllm
date: 2024-04-17 
description: This post is to discuss the swap policy in vllm.
tags: ml
categories: sample-posts
related_posts: false
---

This post is to discuss the swap policy in vllm.
In each step of `llm.engine`, the `scheduler` schedules the blocks to be swapped in and out. Then, the relevant information about the swap-in and swap-out blocks is passed to the `model_executor`
through `scheduler_outputs`.


```python
    def step(self) -> List[RequestOutput]:
        seq_group_metadata_list, scheduler_outputs = self.scheduler.schedule()
        # pdb.set_trace()
        if not scheduler_outputs.is_empty():
            output = self.model_executor.execute_model(
                seq_group_metadata_list, scheduler_outputs.blocks_to_swap_in,
                scheduler_outputs.blocks_to_swap_out,
                scheduler_outputs.blocks_to_copy)
        else:
            output = []
        return self._process_model_outputs(output, scheduler_outputs)
```
The swap-in and swap-out blocks are decided in `scheduler.schedule()` function, which calls `scheduler._schedule()`. The scheduler will try to swap in the blocks from cpu to gpu if 
it is possible. The scheduler will also swap out the blocks from gpu to cpu if `block_manager.can_append_slot` fails. In that case, the scheduler will preempt lower priority seq groups to make space for the higher priority seq groups. In vllm, there are two ways to preempt low priority sequence group: `recompute` and `swap`. According to the code, the current policy is `recompute` only when seq_group.get_max_num_running_seqs() == 1. 
`scheduler.schedule()` will return `scheduler_outputs` which contains necessary information for `model_executor` to do the swap-in and swap-out.
Following are the code showcasing what functions model_executor will call:


1. In `model_executor.execute_model()`, it will call `driver_worker` to do the job.

      ``` python
      output = self.driver_worker.execute_model(
      seq_group_metadata_list=seq_group_metadata_list,
      blocks_to_swap_in=blocks_to_swap_in,
      blocks_to_swap_out=blocks_to_swap_out,
      blocks_to_copy=blocks_to_copy,
    )
      ```

2. In `worker`, it will call `cache_swap` (some codes are ommited)

      ``` python
      if self.is_driver_worker:
      assert seq_group_metadata_list is not None
      num_seq_groups = len(seq_group_metadata_list)
      assert blocks_to_swap_in is not None
      assert blocks_to_swap_out is not None
      assert blocks_to_copy is not None
      data = {
          "num_seq_groups": num_seq_groups,
          "blocks_to_swap_in": blocks_to_swap_in,
          "blocks_to_swap_out": blocks_to_swap_out,
          "blocks_to_copy": blocks_to_copy,
      }
      broadcast_tensor_dict(data, src=0)
      self.cache_swap(blocks_to_swap_in, blocks_to_swap_out, blocks_to_copy)
      ```
3. Then, it will call `cache_engine` to swap the blocks.

      ``` python
        def swap_in(self, src_to_dst: Dict[int, int]) -> None:
            for i in range(self.num_layers):
                self.attn_backend.swap_blocks(self.cpu_cache[i], self.gpu_cache[i],
                                            src_to_dst)
      ```

4. In `attn_backend`(xformer in my case), it will call `Pageattention.swap_blocks` 
      ```python
        def swap_blocks(
            src_kv_cache: torch.Tensor,
            dst_kv_cache: torch.Tensor,
            src_to_dst: Dict[int, int],
        ) -> None:
            PagedAttention.swap_blocks(src_kv_cache, dst_kv_cache, src_to_dst)
      ```
5. And then it will use `cache_ops`.(some codes are ommited) 
      ``` c++
        void swap_blocks(
            torch::Tensor& src,
            torch::Tensor& dst,
            const std::map<int64_t, int64_t>& block_mapping) {
            char *src_ptr = static_cast<char*>(src.data_ptr());
            char *dst_ptr = static_cast<char*>(dst.data_ptr());
            const int64_t block_size_in_bytes = src.element_size() * src[0].numel();
            const at::cuda::OptionalCUDAGuard device_guard(src_device.is_cuda() ? src_device : dst_device);
            const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
            // NOTE(woosuk): This can be slow if the number of blocks is large.
            for (const auto& pair : block_mapping) {
                int64_t src_block_number = pair.first;
                int64_t dst_block_number = pair.second;
                int64_t src_offset = src_block_number * block_size_in_bytes;
                int64_t dst_offset = dst_block_number * block_size_in_bytes;
                cudaMemcpyAsync(
                dst_ptr + dst_offset,
                src_ptr + src_offset,
                block_size_in_bytes,
                memcpy_type,
                stream);
            }
        }
      ```

    The graph might be useful to understand the vllm structure.
<swiper-container keyboard="true" navigation="true" pagination="true" pagination-clickable="true" pagination-dynamic-bullets="true" rewind="true" zoomable=true>
  <swiper-slide>{% include figure.liquid loading="eager" path="assets/img/vllm_structure.jpg" class="img-fluid rounded z-depth-1" %}</swiper-slide>
</swiper-container>


References: [vLL framework](https://zhuanlan.zhihu.com/p/645251151); [vllm](https://github.com/vllm-project/vllm)