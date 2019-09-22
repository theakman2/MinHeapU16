#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

#include <x86intrin.h>

#include <limits>
#include <type_traits>
#include <utility>

/**
 * A fast MinHeap implementation that uses AVX2 instructions for maximum performance.
 * The MinHeap can only store a maximum number of elements, as specified by MaxElementCount. The MinHeap cannot grow.
 */
template <class ValueType, size_t MaxElementCount>
class MinHeapU16 {
	static_assert(std::is_move_assignable<ValueType>::value, "ValueType must be move assignable");

	static_assert(MaxElementCount > 0, "MaxElementCount must be at least 0");

	private:
	static constexpr size_t alignedNumElements = (MaxElementCount + 63ULL) & ~63ULL;

	struct MinCmpResult {
		__m128i m128;

		MinCmpResult() = default;
		MinCmpResult(__m128i x)
			: m128{ x } {
		}

		uint16_t getKey() const {
			return _mm_extract_epi16(m128, 0);
		}

		uint16_t getIndex() const {
			return _mm_extract_epi16(m128, 1);
		}
	};

	uint64_t index{ 0 };
	alignas(32) uint16_t keys[alignedNumElements];
	ValueType values[alignedNumElements];

	public:
	using keyType = uint16_t;
	using valueType = ValueType;
	static constexpr size_t size = MaxElementCount;

	MinHeapU16() {
		std::memset(keys, 0xFF, sizeof(keys));
	}

	/**
	 * Insert a new key/value pair into the heap.
	 * @param key The key to insert. Must not be 65535 (i.e. 0 <= key <= 65534).
	 * @return 0 if insertion succeeded, -1 if there isn't enough space to insert the key/value pair.
	 */
	int insert(uint16_t key, ValueType &&value) {
		if (index == MaxElementCount) {
			return -1;
		}
		uint64_t i = index;
		keys[i] = key;
		values[i] = std::move(value);
		while (i) {
			uint64_t parent = (i - 1) / 64;
			uint16_t pkey = keys[parent];
			if (pkey <= key) {
				break;
			}
			std::swap(keys[parent], keys[i]);
			std::swap(values[parent], values[i]);

			i = parent;
		}
		index++;
		return 0;
	}

	/**
	 * Remove the minimum key/value pair from the heap.
	 * Assumes the heap contains at least one element.
	 * Use `count()` to check the number of elements in the heap.
	 */
	void removeMin() {
		index--;

		keys[0] = keys[index];
		keys[index] = UINT16_MAX;

		std::swap(values[0], values[index]);

		uint64_t i = 0;
		while (1) {
			uint64_t firstChildIndex = i * 64 + 1;
			if (firstChildIndex >= index) {
				break;
			}
			MinCmpResult smallestChild[8];

			const uint16_t *childKeys = keys + firstChildIndex;
			__m128i k0 = _mm_loadu_si128((const __m128i *)(childKeys));
			__m128i k1 = _mm_loadu_si128((const __m128i *)(childKeys + 8));
			__m128i k2 = _mm_loadu_si128((const __m128i *)(childKeys + 16));
			__m128i k3 = _mm_loadu_si128((const __m128i *)(childKeys + 24));
			__m128i k4 = _mm_loadu_si128((const __m128i *)(childKeys + 32));
			__m128i k5 = _mm_loadu_si128((const __m128i *)(childKeys + 40));
			__m128i k6 = _mm_loadu_si128((const __m128i *)(childKeys + 48));
			__m128i k7 = _mm_loadu_si128((const __m128i *)(childKeys + 56));

			smallestChild[0].m128 = _mm_minpos_epu16(k0);
			smallestChild[1].m128 = _mm_minpos_epu16(k1);
			smallestChild[2].m128 = _mm_minpos_epu16(k2);
			smallestChild[3].m128 = _mm_minpos_epu16(k3);
			smallestChild[4].m128 = _mm_minpos_epu16(k4);
			smallestChild[5].m128 = _mm_minpos_epu16(k5);
			smallestChild[6].m128 = _mm_minpos_epu16(k6);
			smallestChild[7].m128 = _mm_minpos_epu16(k7);

			__m128i m128 = _mm_set_epi16(
				smallestChild[7].getKey(),
				smallestChild[6].getKey(),
				smallestChild[5].getKey(),
				smallestChild[4].getKey(),
				smallestChild[3].getKey(),
				smallestChild[2].getKey(),
				smallestChild[1].getKey(),
				smallestChild[0].getKey());
			MinCmpResult u = _mm_minpos_epu16(m128);

			if (u.getKey() >= keys[i]) {
				break;
			}
			uint64_t childIndex = firstChildIndex + u.getIndex() * 8 + smallestChild[u.getIndex()].getIndex();
			keys[childIndex] = keys[i];
			keys[i] = smallestChild[u.getIndex()].getKey();

			std::swap(values[i], values[childIndex]);

			i = childIndex;
		}
	}

	/**
	 * Get the number of elements in the heap.
	 */
	uint64_t count() const {
		return index;
	}

	/**
	 * Fetch the lowest key in the heap without removing it.
	 * Assumes the heap has at least one element.
	 */
	uint16_t peekMinKey() const {
		return keys[0];
	}

	/**
	 * Fetch the value associated with the lowest key in the heap without removing it.
	 * Assumes the heap is not empty.
	 */
	ValueType peekMinValue() const {
		return values[0];
	}

	/**
	 * Fetch the value associated with the lowest key in the heap without removing it.
	 * Assumes the heap is not empty.
	 */
	const ValueType &peekMinValueRef() const {
		return values[0];
	}

	/**
	 * Fetch a pointer to the values in the heap.
	 * Values are stored consecutively in memory.
	 */
	const ValueType *getValues() const {
		return values;
	}

	/**
	 * Fetch a pointer to the values in the heap.
	 * Values are stored consecutively in memory.
	 * Values can be modified in place without having to remove/reinsert the key/value pair.
	 */
	ValueType *getValues() {
		return values;
	}

	/**
	 * Increase all keys in the heap by `amount`.
	 * NOTE: The biggest key must be <= 65534!
	 * You are responsible for ensuring that `amount` doesn't result in any keys exceeding 65534.
	 */
	void increaseAllKeys(uint16_t amount) {
		const __m256i add = _mm256_set1_epi16(amount);

		__m256i *keys256 = reinterpret_cast<__m256i *>(keys);

		uint64_t max = ((index + 63ULL) & ~63ULL) / 16;

		for (uint64_t i = 0; i != max; i += 4) {
			keys256[i] = _mm256_adds_epu16(keys256[i], add);
			keys256[i + 1] = _mm256_adds_epu16(keys256[i + 1], add);
			keys256[i + 2] = _mm256_adds_epu16(keys256[i + 2], add);
			keys256[i + 3] = _mm256_adds_epu16(keys256[i + 3], add);
		}
	}

	/**
	 * Decrease all keys in the heap by `amount` using saturated subtraction.
	 * The operation performed on each key is equivalent to: key = max(0, (int)key - (int)amount)
	 */
	void decreaseAllKeys(uint16_t amount) {
		const __m256i sub = _mm256_set1_epi16(amount);

		__m256i *keys256 = reinterpret_cast<__m256i *>(keys);

		uint64_t max = ((index + 63ULL) & ~63ULL) / 16;

		for (uint64_t i = 0; i != max; i += 4) {
			keys256[i] = _mm256_subs_epu16(keys256[i], sub);
			keys256[i + 1] = _mm256_subs_epu16(keys256[i + 1], sub);
			keys256[i + 2] = _mm256_subs_epu16(keys256[i + 2], sub);
			keys256[i + 3] = _mm256_subs_epu16(keys256[i + 3], sub);
		}

		std::memset(keys + index, 0xFF, (((index + 63U) & ~63U) - index) * sizeof(uint16_t));
	}
};
