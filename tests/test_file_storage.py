import unittest
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from datautils.file_storage import FileStorage

class TestFileStorage(unittest.TestCase):
    def setUp(self):
        # 创建测试用的临时数据目录
        self.test_dir = 'test_data'
        self.storage = FileStorage(self.test_dir)
        
        # 准备测试数据
        self.test_dict = {
            'name': 'test',
            'value': 123,
            'prices': [
                {'date': '2023-01-01', 'price': 100},
                {'date': '2023-01-02', 'price': 101}
            ]
        }
        
        self.test_df = pd.DataFrame({
            'date': ['2023-01-01', '2023-01-02'],
            'value': [1, 2],
            'text': ['a', 'b']
        })

    def tearDown(self):
        # 清理测试数据
        if Path(self.test_dir).exists():
            for file in Path(self.test_dir).glob('*'):
                file.unlink()
            Path(self.test_dir).rmdir()

    def test_json_operations(self):
        # 测试JSON保存和加载
        filename = 'test_json'
        self.assertTrue(self.storage.save_to_json(self.test_dict, filename))
        loaded_data = self.storage.load_from_json(filename)
        self.assertEqual(self.test_dict, loaded_data)

    def test_csv_operations(self):
        # 测试CSV保存和加载
        filename = 'test_csv'
        self.assertTrue(self.storage.save_to_csv(self.test_df, filename))
        loaded_df = self.storage.load_from_csv(filename)
        pd.testing.assert_frame_equal(self.test_df, loaded_df)

    def test_parquet_operations(self):
        # 测试Parquet保存和加载
        filename = 'test_parquet'
        self.assertTrue(self.storage.save_to_parquet(self.test_df, filename))
        loaded_df = self.storage.load_from_parquet(filename)
        pd.testing.assert_frame_equal(self.test_df, loaded_df)

    def test_sql_from_parquet(self):
        # 准备测试数据
        orders_df = pd.DataFrame({
            'order_id': [1, 2, 3],
            'customer_id': [101, 102, 101],
            'amount': [100, 200, 300]
        })
        
        customers_df = pd.DataFrame({
            'customer_id': [101, 102],
            'name': ['Alice', 'Bob']
        })
        
        # 保存测试数据
        self.storage.save_to_parquet(orders_df, 'orders')
        self.storage.save_to_parquet(customers_df, 'customers')
        
        # 准备表路径字典
        table_paths = {
            'orders': str(Path(self.test_dir) / 'orders.parquet'),
            'customers': str(Path(self.test_dir) / 'customers.parquet')
        }
        
        # 测试多表联查
        sql = """
            SELECT o.order_id, c.name, o.amount
            FROM orders o
            JOIN customers c ON o.customer_id = c.customer_id
            WHERE o.amount > 150
        """
        
        result = self.storage.sql_from_parquet(table_paths, sql)
        
        # 预期结果
        expected_df = pd.DataFrame({
            'order_id': [2, 3],
            'name': ['Bob', 'Alice'],
            'amount': [200, 300]
        })
        
        # 确保列顺序一致后再比较
        result = result.sort_values('order_id').reset_index(drop=True)
        expected_df = expected_df.sort_values('order_id').reset_index(drop=True)
        
        # 验证结果
        pd.testing.assert_frame_equal(
            result[expected_df.columns],  # 使用相同的列顺序
            expected_df,
            check_dtype=False  # 添加此参数以忽略数据类型差异
        )

if __name__ == '__main__':
    unittest.main()