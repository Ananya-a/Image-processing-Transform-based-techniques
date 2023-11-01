# import numpy as np
# import pywt
# import cv2

# # Function to extract watermark based on the provided rules
# def extract_watermark(Bw, Pw, size_wtm):
#     u = 0
#     extracted_watermark = []

#     for i in range(2):
#         if u < size_wtm:
#             if Bw[i] > Pw[i]:
#                 extracted_watermark.append(1)
#                 u += 1
#             else:
#                 extracted_watermark.append(0)
#                 u += 1

#     return extracted_watermark

# # Load the watermarked image
# watermarked_image = cv2.imread('watermarked6.jpg', cv2.IMREAD_GRAYSCALE)
# if watermarked_image is None:
#     print("Error: Unable to load the watermarked image.")
# else:
#     # Get the dimensions of the watermarked image
#     height, width = watermarked_image.shape

#     # Initialize an empty list to store the extracted watermark bits
#     extracted_watermark = []

#     # Loop through the watermarked image and extract 8x8 blocks
#     for y in range(0, height, 8):
#         for x in range(0, width, 8):
#             block = watermarked_image[y:y+8, x:x+8]
            
#             # Apply DWT to the block
#             LL, (_, _, _) = pywt.dwt2(block, 'bior1.3')

#             # Calculate the block average
#             block_average = np.mean(LL)

#             # Extract watermark bit based on extraction rules
#             P_w = block_average * 1.0  # Alpha_w should be adjusted based on your needs

#             if watermarked_image[2, 3] > P_w and watermarked_image[3, 2] > P_w:
#                 bit = 255
#             else:
#                 bit = 0

#             extracted_watermark.append(bit)

#     # Convert the list of extracted bits to a NumPy array
#     extracted_watermark = np.array(extracted_watermark)

#     # Reshape the array to match the size of the watermark image
#     watermark_height, watermark_width = watermarked_image.shape # Update to match your watermark size
#     extracted_watermark = extracted_watermark[:watermark_height * watermark_width].reshape((watermark_height, watermark_width))

#     # Convert the binary watermark to an 8-bit grayscale image
#     extracted_watermark_display = extracted_watermark.astype(np.uint8)

#     # Display or save the extracted watermark
#     cv2.imshow("Extracted Watermark", extracted_watermark_display)
#     cv2.imwrite("extracted_watermark.png", extracted_watermark_display)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# # Example usage:
# # Replace with actual DWT-DCT coefficients and values from your process
# Bw = [0.5, 0.8]  # Example coefficients, replace with actual values
# Pw = [0.3, 0.6]  # Example Pw values, replace with actual values
# size_wtm = 8     # Replace with the actual length of the watermark bits

# # Extract the watermark
# extracted_watermark = extract_watermark(Bw, Pw, size_wtm)

# print("Extracted Watermark:", extracted_watermark)







import numpy as np
import pywt
import cv2

# Function to extract watermark based on the provided rules
def extract_watermark(Bw, Pw, size_wtm):
    u = 0
    extracted_watermark = []

    for i in range(2):
        if u < size_wtm:
            if Bw[i] > Pw[i]:
                extracted_watermark.append(1)
                u += 1
            else:
                extracted_watermark.append(0)
                u += 1

    return extracted_watermark

# Load the watermarked image
watermarked_image = cv2.imread('watermarked_img.jpg', cv2.IMREAD_GRAYSCALE)
if watermarked_image is None:
    print("Error: Unable to load the watermarked image.")
else:
    # Get the dimensions of the watermarked image
    height, width = watermarked_image.shape

    # Initialize an empty list to store the extracted watermark bits
    extracted_watermark = []

    # Loop through the watermarked image and extract 8x8 blocks
    for y in range(0, height, 8):
        for x in range(0, width, 8):
            block = watermarked_image[y:y+8, x:x+8]
            
            # Apply DWT to the block
            LL, (_, _, _) = pywt.dwt2(block, 'bior1.3')

            # Calculate the block average
            block_average = np.mean(LL)

            # Extract watermark bit based on extraction rules
            P_w = block_average * 1.0  # Alpha_w should be adjusted based on your needs

            if watermarked_image[2, 3] > P_w and watermarked_image[3, 2] > P_w:
                bit = 255
            else:
                bit = 0

            extracted_watermark.append(bit)

    # Convert the list of extracted bits to a NumPy array
    extracted_watermark = np.array(extracted_watermark)

    # Reshape the array to match the size of the watermark image
    watermark_height, watermark_width = 512, 512  # Update to match your watermark size
    extracted_watermark = extracted_watermark[:watermark_height * watermark_width].reshape((watermark_height, watermark_width))

    # Convert the binary watermark to an 8-bit grayscale image
    extracted_watermark_display = extracted_watermark.astype(np.uint8)

    # Display or save the extracted watermark
    cv2.imshow("Extracted Watermark", extracted_watermark_display)
    cv2.imwrite("extracted_watermark.png", extracted_watermark_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage:
# Replace with actual DWT-DCT coefficients and values from your process
Bw = [0.5, 0.8]  # Example coefficients, replace with actual values
Pw = [0.3, 0.6]  # Example Pw values, replace with actual values
size_wtm = 8     # Replace with the actual length of the watermark bits

# Extract the watermark
extracted_watermark = extract_watermark(Bw, Pw, size_wtm)

print("Extracted Watermark:", extracted_watermark)
