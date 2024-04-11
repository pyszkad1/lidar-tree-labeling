class HistoryQueue:
    def __init__(self, capacity=5):
        self.undoStack = []  # Stack to keep track of history for undo actions
        self.redoStack = []  # Stack to keep redo actions
        self.capacity = capacity

    def push(self, mask_state):
        if len(self.undoStack) >= self.capacity:
            self.undoStack.pop(0)  # Remove the oldest state to maintain capacity
        self.undoStack.append(mask_state)
        self.redoStack.clear()  # Clear the redo stack as new state creates a new branch
        print(f"Pushed state, undoStack: {len(self.undoStack)}, redoStack: {len(self.redoStack)}")

    def undo(self):
        if len(self.undoStack) < 2:
            return None  # Not enough states to undo
        self.redoStack.append(self.undoStack.pop())  # Move the current state to redo stack
        prev_state = self.undoStack[-1]  # Get the state before the last one
        print(f"Undo, undoStack: {len(self.undoStack)}, redoStack: {len(self.redoStack)}")
        return prev_state  # Return the state before the last one

    def redo(self):
        if not self.redoStack:
            return None  # Nothing to redo
        state = self.redoStack.pop()  # Pop the most recent state from redo stack
        self.undoStack.append(state)  # Push it back onto the undo stack
        print(f"Redo, undoStack: {len(self.undoStack)}, redoStack: {len(self.redoStack)}")
        return state
