import pytest
import pandas as pd
from provenquant.core.splitter.sliding_window_split import SlidingWindowSplitter

class TestSlidingWindowSplitter:
  
  def test_basic_split(self):
    """Test basic splitting functionality with simple parameters."""
    splitter = SlidingWindowSplitter(
      n_splits=3,
      train_day=30,
      val_day=10
    )
    
    start_date = pd.Timestamp("2024-01-01")
    end_date = pd.Timestamp("2024-06-01")
    
    splits = list(splitter.split(start_date, end_date))
    assert len(splits) == 3
    
    # Check chronological order
    for i in range(len(splits) - 1):
      assert splits[i][0][0] < splits[i+1][0][0]
      assert splits[i][1][0] < splits[i+1][1][0]
  
  def test_split_with_step_day(self):
    """Test splitting with custom step_day parameter."""
    splitter = SlidingWindowSplitter(
      n_splits=4,
      train_day=30,
      val_day=10,
      step_day=15  # Custom step
    )
    
    start_date = pd.Timestamp("2024-01-01")
    end_date = pd.Timestamp("2024-06-01")
    
    splits = list(splitter.split(start_date, end_date))
    assert len(splits) >= 2
    
    # Check that each split moves by step_day
    if len(splits) >= 2:
      test_end_1 = splits[0][1][1]
      test_end_2 = splits[1][1][1]
      assert (test_end_1 - test_end_2).days == -15
  
  def test_split_with_embargo(self):
    """Test splitting with embargo days."""
    splitter = SlidingWindowSplitter(
      n_splits=2,
      train_day=30,
      val_day=10,
      embargo_day=2
    )
    
    start_date = pd.Timestamp("2024-01-01")
    end_date = pd.Timestamp("2024-06-01")
    
    splits = list(splitter.split(start_date, end_date))
    assert len(splits) >= 1
    
    (train_start, train_end), (test_start, test_end) = splits[0]
    # Verify embargo gap between train_end and test_start
    gap = test_start - train_end
    assert gap.days == 2
  
  def test_split_with_purging(self):
    """Test splitting with purging days."""
    splitter = SlidingWindowSplitter(
      n_splits=2,
      train_day=30,
      val_day=10,
      purging_day=3
    )
    
    start_date = pd.Timestamp("2024-01-01")
    end_date = pd.Timestamp("2024-06-01")
    
    splits = list(splitter.split(start_date, end_date))
    assert len(splits) >= 1
    
    (train_start, train_end), (test_start, test_end) = splits[0]
    gap = test_start - train_end
    assert gap.days == 3
  
  def test_split_with_embargo_and_purging(self):
    """Test splitting with both embargo and purging days."""
    splitter = SlidingWindowSplitter(
      n_splits=2,
      train_day=30,
      val_day=10,
      embargo_day=2,
      purging_day=3
    )
    
    start_date = pd.Timestamp("2024-01-01")
    end_date = pd.Timestamp("2024-06-01")
    
    splits = list(splitter.split(start_date, end_date))
    assert len(splits) >= 1
    
    (train_start, train_end), (test_start, test_end) = splits[0]
    gap = test_start - train_end
    assert gap.days == 5  # embargo + purging
  
  def test_no_splits_when_insufficient_data(self):
    """Test that no splits are returned when insufficient data."""
    splitter = SlidingWindowSplitter(
      n_splits=10,
      train_day=100,
      val_day=10
    )
    
    start_date = pd.Timestamp("2024-01-01")
    end_date = pd.Timestamp("2024-01-10")
    
    splits = list(splitter.split(start_date, end_date))
    assert len(splits) == 0
  
  def test_validation_period_length(self):
    """Test that validation period has correct length."""
    splitter = SlidingWindowSplitter(
      n_splits=2,
      train_day=30,
      val_day=7
    )
    
    start_date = pd.Timestamp("2024-01-01")
    end_date = pd.Timestamp("2024-06-01")
    
    splits = list(splitter.split(start_date, end_date))
    assert len(splits) >= 1
    
    (_, _), (test_start, test_end) = splits[0]
    assert (test_end - test_start).days == 7
  
  def test_training_period_length(self):
    """Test that training period has correct length."""
    splitter = SlidingWindowSplitter(
      n_splits=2,
      train_day=15,
      val_day=5
    )
    
    start_date = pd.Timestamp("2024-01-01")
    end_date = pd.Timestamp("2024-06-01")
    
    splits = list(splitter.split(start_date, end_date))
    assert len(splits) >= 1
    
    (train_start, train_end), (_, _) = splits[0]
    assert (train_end - train_start).days == 15
  
  def test_large_date_range_multiple_splits(self):
    """Test with large date range similar to user's example."""
    splitter = SlidingWindowSplitter(
      n_splits=4,
      train_day=365,
      val_day=30,
      embargo_day=1,
      purging_day=1
    )
    
    start_date = pd.Timestamp("2024-01-01")
    end_date = pd.Timestamp("2026-01-27")
    
    splits = list(splitter.split(start_date, end_date))
    
    # Should get 4 splits with this date range
    assert len(splits) == 4
    
    # Verify all periods have correct lengths
    for (train_start, train_end), (test_start, test_end) in splits:
      assert (train_end - train_start).days == 365
      assert (test_end - test_start).days == 30
      assert (test_start - train_end).days == 2  # embargo + purging
  
  def test_chronological_order_of_splits(self):
    """Test that splits are returned in chronological order."""
    splitter = SlidingWindowSplitter(
      n_splits=5,
      train_day=30,
      val_day=10,
      step_day=10
    )
    
    start_date = pd.Timestamp("2024-01-01")
    end_date = pd.Timestamp("2024-12-31")
    
    splits = list(splitter.split(start_date, end_date))
    
    # Check that each subsequent split starts after the previous
    for i in range(len(splits) - 1):
      current_train_start = splits[i][0][0]
      next_train_start = splits[i + 1][0][0]
      assert next_train_start > current_train_start
  
  def test_no_overlap_between_train_and_test(self):
    """Test that train and test periods don't overlap."""
    splitter = SlidingWindowSplitter(
      n_splits=3,
      train_day=30,
      val_day=10,
      embargo_day=1
    )
    
    start_date = pd.Timestamp("2024-01-01")
    end_date = pd.Timestamp("2024-12-31")
    
    splits = list(splitter.split(start_date, end_date))
    
    for (train_start, train_end), (test_start, test_end) in splits:
      # Test should start after train ends
      assert test_start >= train_end
  
  def test_default_step_day_equals_val_day(self):
    """Test that default step_day is equal to val_day."""
    splitter = SlidingWindowSplitter(
      n_splits=3,
      train_day=30,
      val_day=15
    )
    
    start_date = pd.Timestamp("2024-01-01")
    end_date = pd.Timestamp("2024-12-31")
    
    splits = list(splitter.split(start_date, end_date))
    
    if len(splits) >= 2:
      # Check that test periods move by val_day (default step)
      test_end_1 = splits[0][1][1]
      test_end_2 = splits[1][1][1]
      assert (test_end_1 - test_end_2).days == -15