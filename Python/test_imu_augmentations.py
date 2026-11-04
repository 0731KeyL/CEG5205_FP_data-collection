import unittest

import numpy as np

from imu_augmentations import (
    add_packet_loss,
    add_sensor_outage,
    augment_window,
    crop_and_resize,
    resample_sequence,
    shift_with_padding,
)


class AugmentationTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(7)
        self.values = np.arange(120, dtype=np.float32).reshape(10, 12)
        self.mask = np.ones_like(self.values, dtype=bool)

    def test_resample_preserves_channels_and_endpoints(self):
        result = resample_sequence(self.values, 25)
        self.assertEqual(result.shape, (25, 12))
        np.testing.assert_allclose(result[0], self.values[0])
        np.testing.assert_allclose(result[-1], self.values[-1])

    def test_crop_changes_length_back_to_target(self):
        result, fraction = crop_and_resize(self.values, 50, self.rng)
        self.assertEqual(result.shape, (50, 12))
        self.assertGreaterEqual(fraction, 0.70)
        self.assertLessEqual(fraction, 1.0)

    def test_positive_shift_marks_leading_samples_invisible(self):
        shifted, mask = shift_with_padding(
            self.values,
            self.mask,
            shift=2,
        )
        np.testing.assert_allclose(shifted[:2], 0)
        self.assertFalse(mask[:2].any())
        self.assertTrue(mask[2:].all())

    def test_packet_loss_uses_hold_value_and_mask(self):
        values, mask, added = add_packet_loss(
            self.values,
            self.mask,
            np.random.default_rng(1),
            bursts=5,
            max_burst=2,
        )
        self.assertGreaterEqual(added, 1)
        self.assertTrue(np.isfinite(values).all())
        self.assertTrue((~mask).any())

    def test_sensor_outage_only_drops_one_six_channel_group(self):
        values, mask, sensor = add_sensor_outage(
            self.values,
            self.mask,
            np.random.default_rng(3),
        )
        self.assertIsNotNone(sensor)
        hidden_channels = slice(sensor * 6, sensor * 6 + 6)
        other_channels = slice((1 - sensor) * 6, (1 - sensor) * 6 + 6)
        hidden_rows = np.where((~mask[:, hidden_channels]).all(axis=1))[0]
        self.assertGreater(len(hidden_rows), 0)
        self.assertTrue(mask[:, other_channels].all())
        self.assertTrue(np.isfinite(values).all())

    def test_full_augmentation_has_fixed_shape_and_visible_data(self):
        result, mask, info = augment_window(
            self.values,
            target_length=50,
            seed=11,
        )
        self.assertEqual(result.shape, (50, 12))
        self.assertEqual(mask.shape, (50, 12))
        self.assertTrue(np.isfinite(result).all())
        self.assertTrue(mask.any())
        self.assertGreaterEqual(info.shift_samples, -10)
        self.assertLessEqual(info.shift_samples, 10)


if __name__ == "__main__":
    unittest.main()
