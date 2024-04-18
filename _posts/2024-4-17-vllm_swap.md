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
