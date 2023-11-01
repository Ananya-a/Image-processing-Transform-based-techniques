# import cv2
# import numpy as np
# import pywt

# # Load the watermarked image
# watermarked_image = cv2.imread('watermarked.jpg', cv2.IMREAD_GRAYSCALE)
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

#             # Extract watermark bit (assuming 0 or 255 for binary watermark)
#             if LL[2, 2] > block_average:
#                 bit = 255
#             else:
#                 bit = 0
#             extracted_watermark.append(bit)

#     # Convert the list of extracted bits to a NumPy array
#     extracted_watermark = np.array(extracted_watermark)

#     # Calculate the size of the extracted watermark
#     extracted_watermark_size = len(extracted_watermark)

#     # If you know the dimensions of the original watermark image, you can reshape it accordingly
#     # For example, if it's 32x32:
#     watermark_height, watermark_width = 32, 32
#     extracted_watermark = extracted_watermark[:watermark_height * watermark_width].reshape((watermark_height, watermark_width))

#     # Convert the binary watermark to an 8-bit grayscale image
#     extracted_watermark_display = np.uint8(extracted_watermark)

#     # Display or save the extracted watermark
#     cv2.imshow("Extracted Watermark", extracted_watermark_display)
#     cv2.imwrite("extracted_watermark.png", extracted_watermark_display)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


# import cv2
# import numpy as np
# import pywt

# # Load the watermarked image
# watermarked_image = cv2.imread('watermarked.jpg', cv2.IMREAD_GRAYSCALE)
# if watermarked_image is None:
#     print("Error: Unable to load the watermarked image.")
# else:
#     # Get the dimensions of the watermarked image
#     height, width = watermarked_image.shape

#     # Initialize variables
#     watermark_size = 32 * 32  # Update the size based on your watermark size
#     extracted_watermark = np.zeros(watermark_size, dtype=np.uint8)

#     # Loop through the watermarked image and extract watermark bits using the specified rules
#     u = 0
#     for i in range(2):
#         if u < watermark_size:
#             LL, (_, _, _) = pywt.dwt2(watermarked_image, 'bior1.3')
#             block_average = np.mean(LL)
#             alpha_w = 0.5  # Adjust alpha_w as needed
#             P_w = block_average * alpha_w

#             if watermarked_image[2, 3] > P_w and u < watermark_size:
#                 extracted_watermark[u] = 255
#                 u += 1
#             else:
#                 extracted_watermark[u] = 0
#                 u += 1

#     # Reshape the extracted watermark to its original size
#     extracted_watermark = extracted_watermark.reshape((32, 32))

#     # Display or save the extracted watermark
#     cv2.imshow("Extracted Watermark", extracted_watermark)
#     cv2.imwrite("extracted_watermark.png", extracted_watermark)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


import cv2
import numpy as np
import pywt

# Load the watermarked image
watermarked_image = cv2.imread('watermarked6.jpg', cv2.IMREAD_GRAYSCALE)
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
    watermark_height, watermark_width = watermarked_image.shape # Update to match your watermark size
    extracted_watermark = extracted_watermark[:watermark_height * watermark_width].reshape((watermark_height, watermark_width))

    # Convert the binary watermark to an 8-bit grayscale image
    extracted_watermark_display = extracted_watermark.astype(np.uint8)

    # Display or save the extracted watermark
    cv2.imshow("Extracted Watermark", extracted_watermark_display)
    cv2.imwrite("extracted_watermark.png", extracted_watermark_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
