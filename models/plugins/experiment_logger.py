import json
import numpy as np

class ExperimentLogger:
    def __init__(self, log_file_path):
        """Initializes the logger."""
        try:
            self.log_file = open(log_file_path, 'w')
        except IOError as e:
            print(f"Error opening log file {log_file_path}: {e}")
            self.log_file = None

    def _convert_numpy_types(self, obj):
        """递归转换numpy类型为Python原生类型，确保JSON序列化安全"""
        if isinstance(obj, dict):
            return {k: self._convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._convert_numpy_types(item) for item in obj)
        elif hasattr(obj, 'item'):  # numpy标量类型
            return obj.item()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    def log_step(self, data_dict):
        """Records a single step of data in JSONL format."""
        if self.log_file:
            try:
                # 转换numpy类型为Python原生类型
                safe_data = self._convert_numpy_types(data_dict)
                self.log_file.write(json.dumps(safe_data) + '\n')
                self.log_file.flush()
            except Exception as e:
                # 如果仍然失败，记录错误但不中断程序
                print(f"[ExperimentLogger] JSON serialization failed: {e}")
                print(f"[ExperimentLogger] Problematic data keys: {list(data_dict.keys())}")
                # 写入错误信息而不是原始数据
                error_data = {"error": f"JSON serialization failed: {str(e)}", "keys": list(data_dict.keys())}
                self.log_file.write(json.dumps(error_data) + '\n')
                self.log_file.flush()

    def close(self):
        """Closes the log file."""
        if self.log_file:
            self.log_file.close() 