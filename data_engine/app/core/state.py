from typing import List, Optional
import pandas as pd

# Global State
active_df: Optional[pd.DataFrame] = None
active_file_path: Optional[str] = None
history_stack: List[pd.DataFrame] = []
redo_stack: List[pd.DataFrame] = []
action_history: List[str] = []
redo_action_history: List[str] = []
active_model = None
active_model_metadata: Optional[dict] = None

def get_active_df():
    global active_df
    return active_df

def set_active_df(df: pd.DataFrame):
    global active_df
    active_df = df

def get_history_stack():
    return history_stack

def push_to_history(df: pd.DataFrame):
    global history_stack
    history_stack.append(df)
    if len(history_stack) > 10:
        history_stack.pop(0)


def get_redo_stack():
    return redo_stack

def get_action_history():
    return action_history

def get_redo_action_history():
    return redo_action_history

def get_active_model():
    global active_model
    return active_model

def set_active_model(model):
    global active_model
    active_model = model

def get_active_model_metadata():
    global active_model_metadata
    return active_model_metadata

def set_active_model_metadata(metadata: dict):
    global active_model_metadata
    active_model_metadata = metadata

def reset_state(new_file_path: str = None):
    global active_file_path, history_stack, redo_stack, action_history, redo_action_history, active_model, active_model_metadata
    active_file_path = new_file_path
    history_stack.clear()
    redo_stack.clear()
    action_history.clear()
    redo_action_history.clear()
    active_model = None
    active_model_metadata = None
