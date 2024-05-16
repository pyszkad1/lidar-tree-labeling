class HistoryQueue:
    def __init__(self, capacity=5):
        self.undoStack = []
        self.redoStack = []
        self.capacity = capacity

    def push(self, mask_state):
        if len(self.undoStack) >= self.capacity:
            self.undoStack.pop(0)
        self.undoStack.append(mask_state)
        self.redoStack.clear()
        print(f"Pushed state, undoStack: {len(self.undoStack)}, redoStack: {len(self.redoStack)}")

    def undo(self):
        if len(self.undoStack) < 2:
            return None
        self.redoStack.append(self.undoStack.pop())
        prev_state = self.undoStack[-1]
        print(f"Undo, undoStack: {len(self.undoStack)}, redoStack: {len(self.redoStack)}")
        return prev_state

    def redo(self):
        if not self.redoStack:
            return None
        state = self.redoStack.pop()
        self.undoStack.append(state)
        print(f"Redo, undoStack: {len(self.undoStack)}, redoStack: {len(self.redoStack)}")
        return state
