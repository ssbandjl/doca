/*
 * Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * This software product is a proprietary product of NVIDIA CORPORATION &
 * AFFILIATES (the "Company") and all right, title, and interest in and to the
 * software product, including all associated intellectual property rights, are
 * and shall remain exclusively with the Company.
 *
 * This software product is governed by the End User License Agreement
 * provided with the software product.
 */

#ifndef __VIRTNET_DPA_STACK_H__
#define __VIRTNET_DPA_STACK_H__

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct stack {
       void **arr;
       int size;
       int top;
};

#define STACK_ENTRY_INVALID -1
static inline void stack_init(struct stack *s, void *arr, int size)
{
	s->arr = arr;
	s->size = size;
	s->top = STACK_ENTRY_INVALID;
};

static inline bool stack_empty(const struct stack *s)
{
	return (s->top == STACK_ENTRY_INVALID) ? true : false;
};

static inline bool stack_full(const struct stack *s)
{
	return (s->top == s->size - 1) ? true : false;
};

static inline void *stack_pop(struct stack *s)
{
	if (stack_empty(s))
		return NULL;

	return s->arr[s->top--];
};

static inline void stack_push(struct stack *s, void *e)
{
	if (stack_full(s))
		return;

	s->arr[++s->top] = e;
};

/* don't check if stack is full to save cycles */
static inline void stack_push_no_check(struct stack *s, void *e)
{
	s->arr[++s->top] = e;
};

#ifdef __cplusplus
}
#endif

#endif
