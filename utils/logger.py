# utils/logger.py
import os
import sys


class DefaultLogger:
    """
    Simple file logger to record experiment metrics.
    Writes metrics to a text file in real-time.
    """

    def __init__(self, exp_name, log_dir):
        self.exp_name = exp_name
        self.log_dir = log_dir

        # Ensure directory exists
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        self.file_path = os.path.join(log_dir, exp_name)
        self.file = open(self.file_path, 'w')
        print(f"[Logger] Log file created at: {self.file_path}")

    def close(self):
        if self.file:
            self.file.close()

    def _write_line(self, text):
        """Helper to write and flush immediately."""
        self.file.write(text + '\n')
        self.file.flush()  # Crucial for saving logs before a potential crash

    def write_round(self, round_idx):
        self._write_line(f"round : {round_idx}")

    def write_test_acc(self, acc):
        self._write_line(f"global_test_acc : {acc:.4f}")

    def write_test_loss(self, loss):
        self._write_line(f"test_loss : {loss:.4f}")

    def write_test_f1(self, f1):
        """[NEW] Log Macro-F1 Score"""
        self._write_line(f"macro_f1 : {f1:.4f}")

    def write_mean_val_acc(self, mean_val_acc):
        self._write_line(f"client_mean_val_acc : {mean_val_acc:.4f}")

    def write_std_val_loss(self, std_val_loss):
        self._write_line(f"client_std_val_loss : {std_val_loss:.4f}")

    def write_mean_val_loss(self, mean_val_loss):
        self._write_line(f"client_mean_val_loss : {mean_val_loss:.4f}")

    def write_std_val_acc(self, std_val_acc):
        self._write_line(f"client_std_val_acc : {std_val_acc:.4f}")