import unittest
import torch
from src.gcnn import CyclicGroup, DihedralGroup


class TestCyclicGroup(unittest.TestCase):
    def setUp(self):
        self.c4 = CyclicGroup(order=4)
        self.elements = self.c4.elements()
        self.e, self.g1, self.g2, self.g3 = self.elements

    def test_elements(self):
        self.assertEqual(len(self.elements), 4)

    def test_product(self):
        self.assertTrue(torch.allclose(self.c4.product(self.e, self.g1), self.g1))
        self.assertTrue(torch.allclose(self.c4.product(self.g1, self.g2), self.g3))
        self.assertTrue(
            torch.allclose(self.c4.product(self.g1, self.c4.inverse(self.g1)), self.e)
        )

        group_elements = self.c4.elements()
        inv_group_elements = torch.stack(
            [self.c4.inverse(group_element) for group_element in group_elements]
        )
        product = torch.stack(
            [
                self.c4.product(inv_element, group_element)
                for inv_element, group_element in zip(
                    inv_group_elements, group_elements
                )
            ]
        )
        self.assertTrue(torch.allclose(product, self.e))

    def test_inverse(self):
        inv_g1 = self.c4.inverse(self.g1)
        self.assertTrue(torch.allclose(self.c4.product(self.g1, inv_g1), self.e))

    def test_matrix_representation(self):
        identity = torch.eye(2)
        mat_e = self.c4.matrix_representation(self.e)
        mat_g2 = self.c4.matrix_representation(self.g2)
        self.assertTrue(torch.allclose(mat_e, identity, atol=1e-6))
        minus_I = torch.tensor([[-1, 0], [0, -1]], dtype=torch.float32)
        self.assertTrue(torch.allclose(mat_g2, minus_I, atol=1e-6))

    def test_left_action_on_R2(self):
        v = torch.tensor([0.0, 1.0])
        expected = torch.tensor([-1.0, 0.0])
        result = self.c4.left_action_on_R2(self.g1, v)
        self.assertTrue(torch.allclose(result, expected, atol=1e-6))


class TestDihedralGroup(unittest.TestCase):
    def setUp(self):
        self.d4 = DihedralGroup(polygon_sides=4)
        self.elements = self.d4.elements()
        self.rot0 = self.elements[0]  # [0, 0]
        self.ref0 = self.elements[4]  # [0, 1]

    def test_elements(self):
        self.assertEqual(self.elements.shape, (8, 2))

    def test_product(self):
        prod_rot = self.d4.product(self.rot0, self.rot0)
        prod_ref = self.d4.product(self.ref0, self.ref0)
        self.assertTrue(torch.allclose(prod_rot, self.rot0))
        self.assertTrue(torch.allclose(prod_ref, self.rot0))

        # Additional products
        rot1 = self.elements[1]  # [theta1, 0]
        ref1 = self.elements[5]  # [theta1, 1]
        # rotation * reflection
        prod_rot_ref = self.d4.product(self.rot0, ref1)
        self.assertTrue(torch.allclose(prod_rot_ref, ref1))
        # reflection * rotation
        prod_ref_rot = self.d4.product(self.ref0, rot1)
        expected = self.d4.product(self.ref0, rot1)
        self.assertTrue(torch.allclose(prod_ref_rot, expected))
        # reflection * reflection (different)
        prod_ref0_ref1 = self.d4.product(self.ref0, ref1)
        expected_rot = self.d4.product(self.ref0, ref1)
        self.assertTrue(torch.allclose(prod_ref0_ref1, expected_rot))

        group_elements = self.d4.elements()
        inv_group_elements = torch.stack(
            [self.d4.inverse(group_element) for group_element in group_elements]
        )
        product = torch.stack(
            [
                self.d4.product(inv_element, group_element)
                for inv_element, group_element in zip(
                    inv_group_elements, group_elements
                )
            ]
        )
        self.assertTrue(torch.allclose(product, self.rot0))

    def test_inverse(self):
        inv_rot = self.d4.inverse(self.rot0)
        inv_ref = self.d4.inverse(self.ref0)
        self.assertTrue(torch.allclose(self.d4.product(self.rot0, inv_rot), self.rot0))
        self.assertTrue(torch.allclose(self.d4.product(self.ref0, inv_ref), self.rot0))

    def test_matrix_representation(self):
        mat_rot = self.d4.matrix_representation(self.rot0)
        identity = torch.eye(2)
        mat_ref = self.d4.matrix_representation(self.ref0)
        self.assertTrue(torch.allclose(mat_rot, identity, atol=1e-6))
        self.assertTrue(torch.allclose(mat_ref, identity, atol=1e-6))

    def test_left_action_on_R2(self):
        v = torch.tensor([1.0, 0.0])
        result = self.d4.left_action_on_R2(self.rot0, v)
        self.assertTrue(torch.allclose(result, v, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
