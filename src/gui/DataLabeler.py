import cv2
import numpy as np

# Dummy data loader for color images
def load_color_image_data():
    # Example: Generate a dummy color image (200x200) as a numpy array
    return np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)

class ImageLabeler:
    def __init__(self, images):
        self.images = images
        self.index = 0
        self.drawing = False # True if mouse is pressed
        self.mask = None

        # Create window and set a mouse callback
        cv2.namedWindow('ImageLabeler')
        cv2.setMouseCallback('ImageLabeler', self.draw)

    def draw(self, event, x, y, flags, param):
        # Mouse callback function for drawing
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing == True:
                cv2.circle(self.mask, (x, y), 5, (1, 1, 1), -1) # Draw with 1's on the mask
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            cv2.circle(self.mask, (x, y), 5, (1, 1, 1), -1)

    def run(self):
        # Main loop
        mask_dir_path = 'masks'
        for img in self.images:
            self.mask = np.zeros(img.shape[:2], dtype=np.uint8) # Initialize mask for current image

            while True:
                overlay = img.copy()
                overlay[self.mask == 1] = [0, 255, 255] # Make the masked area yellow
                cv2.imshow('ImageLabeler', overlay) # Display the image with drawing overlay

                k = cv2.waitKey(1) & 0xFF
                if k == ord('n'): # Next image
                    break
                elif k == ord('q'): # Quit
                    cv2.destroyAllWindows()
                    return

                elif k == ord('s'):  # Save mask
                    mask_file_path = f'{mask_dir_path}/mask_{self.index}.txt'
                    np.savetxt(mask_file_path, self.mask, fmt='%d')

                    print(f'Mask saved to {mask_file_path}')

            mask_file_path = f'{mask_dir_path}/mask_{self.index}.txt'
            np.savetxt(mask_file_path, self.mask, fmt='%d')
            print(f'Mask saved to {mask_file_path}')
            self.index += 1

            # Process/save the mask for the current image here
            # For example, you can save the mask to a file or add it to a list

if __name__ == "__main__":
    # Example usage
    images = [load_color_image_data() for _ in range(5)] # Load your images as numpy arrays
    labeler = ImageLabeler(images)
    labeler.run()
