from datetime import timedelta
import numpy as np

class SlidingWindowSplitter:
    def __init__(
        self,
        n_splits: int,
        train_day: int,
        val_day: int,
        embargo_day: int = 0,
        purging_day: int = 0,
        step_day: int = None
    ):
        self.n_splits = n_splits
        self.train_day = train_day
        self.val_day = val_day
        self.embargo_day = embargo_day
        self.purging_day = purging_day
        # Default step is validation period
        self.step_day = step_day if step_day is not None else val_day
    
    def split(self, start_date, end_date):
        """
        Split data into walk-forward train/test periods.
        Starts from end_date and calculates backwards, then returns splits in chronological order.
        
        Args:
            start_date: Start date for splitting
            end_date: End date for splitting
            
        Yields:
            ((train_start, train_end), (test_start, test_end)) tuples
        """
        splits = []
        current_test_end = end_date
        
        for _ in range(self.n_splits):
            test_end = current_test_end
            test_start = test_end - timedelta(days=self.val_day)
            train_end = test_start - timedelta(days=self.embargo_day + self.purging_day)
            train_start = train_end - timedelta(days=self.train_day)
            
            if train_start < start_date:
                break
            
            splits.append(((train_start, train_end), (test_start, test_end)))
            # Move backward by step_day instead of entire window
            current_test_end = test_end - timedelta(days=self.step_day)
        
        # Return splits in chronological order
        for split in reversed(splits):
            yield split
