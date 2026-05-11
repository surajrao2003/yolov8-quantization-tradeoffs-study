import cv2
import numpy as np

def resize_and_pad_frame(frame, new_shape=(640, 640), auto=False, scaleFill=False, scaleup=True, stride=32):
    shape = frame.shape[:2] # Get the shape of the input frame
    # Convert new_shape to tuple if it's an integer
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])     # Calculate the resizing ratio
    # Ensure scaleup behavior based on the scaleup flag
    if not scaleup:
        r = min(r, 1.0)
    ratio = r, r  # Compute the resizing ratio for both dimensions
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r))) # Calculate the new dimensions after resizing (unpadded)
    # Calculate the padding sizes
    dw = (new_shape[1] - new_unpad[0]) / 2
    dh = (new_shape[0] - new_unpad[1]) / 2
    # Ensure integer values for padding
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    color = [114, 114, 114] # Define the padding color 
    new_img = cv2.resize(frame, new_unpad, interpolation=cv2.INTER_LINEAR) # Resize the frame using linear interpolation
    new_img = cv2.copyMakeBorder(new_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color) # Apply padding to the resized frame
    # Return the resized and padded image along with resizing and padding ratios
    return new_img, (r, r), (dw, dh)

def preprocess_frame(
    input_frame, input_size=(640, 640), *, feed_dtype=np.float32
):
    """
    feed_dtype: ONNX Runtime input tensor element type from the loaded model's
    "images" input (typically float32; use float16 only if ONNX declares tensor(float16)).
    """
    original_frame_shape = input_frame.shape[:2]  # Store original frame shape
    img = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB) # Convert BGR to RGB color space
    img, scale, pad = resize_and_pad_frame(img, new_shape=input_size)  # Resize and pad the frame , return scale and pad
    frame_data = np.array(img) / 255.0 # Normalize pixel values to the range [0, 1]
    frame_data = np.transpose(frame_data, (2, 0, 1))  # Reorder dimensions to have channels first
    frame_data = np.expand_dims(frame_data, axis=0).astype(feed_dtype)  # Add batch dimension + match ONNX I/O dtype
    frame_data = np.ascontiguousarray(frame_data)  # Ensure contiguous memory layout
    return frame_data, input_frame, original_frame_shape, scale, pad