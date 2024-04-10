class HistoryQueue:
    def __init__(self, capacity=5):
        self.capacity = capacity
        self.queue = [None] * capacity
        self.current = -1  # Points to the current state
        self.oldest = 0  # Points to the oldest state

    def push(self, mask_state):
        # Advance and wrap the current pointer
        self.current = (self.current + 1) % self.capacity
        # If the queue is full, oldest moves with current
        if self.current == self.oldest:
            self.oldest = (self.oldest + 1) % self.capacity
        self.queue[self.current] = mask_state

    def undo(self):
        if self.current == self.oldest and self.queue[self.current] is None:
            return None  # Nothing to undo
        prev_state = self.queue[self.current]
        self.current = (self.current - 1) % self.capacity
        if self.queue[self.current] is None:  # Fix wrapping issue
            self.current = (self.capacity - 1)
        return prev_state