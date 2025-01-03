from . import config as conf

class CoverageFilter:
    def __init__(self):
        # Similar to std::array<uint8_t, BLOOM_FILTER_SIZE_BYTES> filled with 0
        self.bits_ = bytearray(conf.BLOOM_FILTER_SIZE_BYTES)

    def add(self, item: int) -> None:
        """Add an item to the Bloom filter."""
        idx1 = self.hash1(item)
        idx2 = self.hash2(item)

        self._set_bit(idx1)
        self._set_bit(idx2)

    def check(self, item: int) -> bool:
        """
        Check if an item might be in the Bloom filter.
        Returns False if definitely not in the filter,
        True if possibly in (false positives possible).
        """
        idx1 = self.hash1(item)
        if not self._test_bit(idx1):
            return False

        idx2 = self.hash2(item)
        if not self._test_bit(idx2):
            return False

        return True

    def merge(self, other: "CoverageFilter") -> None:
        """Merge another CoverageFilter into this one (bitwise OR)."""
        for i in range(conf.BLOOM_FILTER_SIZE_BYTES):
            self.bits_[i] |= other.bits_[i]

    def clear(self) -> None:
        """Clear all bits."""
        for i in range(conf.BLOOM_FILTER_SIZE_BYTES):
            self.bits_[i] = 0

    def __str__(self) -> str:
        """Return a human-readable representation of the filter's bits."""
        return f"CoverageFilter(0x{self.bits_.hex()})"

    # ------------------------
    # Private / Helper Methods
    # ------------------------

    def _set_bit(self, index: int) -> None:
        """Set the bit at a specific index."""
        if index >= conf.BLOOM_FILTER_SIZE_BITS:
            return  # Out-of-range check
        byte_index = index // 8
        bit_mask = 1 << (index % 8)
        self.bits_[byte_index] |= bit_mask

    def _test_bit(self, index: int) -> bool:
        """Test (check) the bit at a specific index."""
        if index >= conf.BLOOM_FILTER_SIZE_BITS:
            return False
        byte_index = index // 8
        bit_mask = 1 << (index % 8)
        return (self.bits_[byte_index] & bit_mask) != 0

    def hash1(self, value: int) -> int:
        """
        Very simplistic hash function inspired by the C++ code:
          combined = value ^ (seed1 + (value << 6) + (value >> 2))
          then apply std::hash (approximated by Python's built-in hash).
        """
        seed1 = 0xDEADBEEF
        combined = value ^ (seed1 + (value << 6) + (value >> 2))
        # Python's hash can be negative; mask to 64-bit to mimic unsigned behavior
        hash_out = hash(combined) & ((1 << 64) - 1)
        return hash_out % conf.BLOOM_FILTER_SIZE_BITS

    def hash2(self, value: int) -> int:
        """
        Another simplistic hash function inspired by the C++ code:
          combined = value ^ (seed2 + (value << 5) + (value >> 3))
        """
        seed2 = 0xBADC0FFE
        combined = value ^ (seed2 + (value << 5) + (value >> 3))
        # Python's hash can be negative; mask to 64-bit
        hash_out = hash(combined) & ((1 << 64) - 1)
        return hash_out % conf.BLOOM_FILTER_SIZE_BITS
