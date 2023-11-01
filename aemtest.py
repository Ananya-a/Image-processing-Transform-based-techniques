# import cv2
# import numpy as np
# import pywt

# # Load the host image
# host_image = cv2.imread('lenna.png', cv2.IMREAD_GRAYSCALE)
# if host_image is None:
#     print("Error: Unable to load the image.")
# else:
#     watermark_image = cv2.imread('Binary2.png', cv2.IMREAD_GRAYSCALE)

#     if watermark_image is None:
#         print("Error: Unable to load the watermark image.")
#     else:
#         # Get the dimensions of the host image
#         height, width = host_image.shape

#         # Check if the dimensions are divisible by 8 (for simplicity)
#         if height % 8 != 0 or width % 8 != 0:
#             print("Image dimensions are not divisible by 8. Adjust the image size.")
#         else:
#             # Resize the watermark image to match the size of LL subband (4x4)
#             watermark_image = cv2.resize(watermark_image, (32, 32))

#             # Initialize an empty list to store the LL bands after DWT
#             ll_bands = []

#             # Loop through the image and extract 8x8 blocks
#             for y in range(0, height, 8):
#                 for x in range(0, width, 8):
#                     block = host_image[y:y+8, x:x+8]
#                     # Apply DWT to the block
#                     LL, (LH, HL, HH) = pywt.dwt2(block, 'bior1.3')
#                     # Calculate the values required for watermark embedding rules
#                     alpha = (HL[2, 2] + LH[2, 2]) / 16.0
#                     #alpha=0.5
#                     block_average = np.mean(LL)
#                     P = block_average / alpha
#                     # Apply watermark embedding rules here
#                     for i in range(4):
#                         for j in range(4):
#                             if (i * 4 + j) < watermark_image.size:
#                                 bit = watermark_image[i, j]  # Assuming watermark_image is a binary image
#                                 if bit == 0 and (P - LL[i, j]) > alpha:
#                                     LL[i, j] = P - alpha
#                                 elif bit == 1 and (LL[i, j] - P) > alpha:
#                                     LL[i, j] = P + alpha

#                     # Reconstruct the watermarked block
#                     watermarked_block = pywt.idwt2((LL, (LH, HL, HH)), 'bior1.3')
#                     ll_bands.append(watermarked_block)

#             # Assemble the watermarked 8x8 blocks into the original image
#             watermarked_image = np.zeros((height, width), dtype=np.uint8)
#             block_idx = 0

#             for y in range(0, height, 8):
#                 for x in range(0, width, 8):
#                     if block_idx < len(ll_bands):
#                         watermarked_image[y:y+8, x:x+8] = ll_bands[block_idx]
#                         block_idx += 1

#             # Display the host image with the watermark embedded
#             cv2.imshow("Host Image with Watermark", watermarked_image)
#             cv2.imwrite("watermarked.jpg",watermarked_image)
#             cv2.waitKey(0)
#             cv2.destroyAllWindows()



import cv2
import numpy as np
import pywt

# Load the host image
host_image = cv2.imread('lenna.png', cv2.IMREAD_GRAYSCALE)
original_image = cv2.imread('lenna.png', cv2.IMREAD_GRAYSCALE)
watermark_image = cv2.imread('Binary2.png', cv2.IMREAD_GRAYSCALE)

if watermark_image is None:
        print("Error: Unable to load the watermark image.")
else:
        # Get the dimensions of the host image
        height, width = host_image.shape

        # Check if the dimensions are divisible by 8 (for simplicity)
        if height % 8 != 0 or width % 8 != 0:
            print("Image dimensions are not divisible by 8. Adjust the image size.")
        else:
            # Initialize an empty list to store the LL bands after DWT
            ll_bands = []
            blocks=[]
            variances2=[]
            for i in range(0, 512, 8):
                for j in range(0, 512, 8):
                    block = host_image[i:i+8, j:j+8]
                    blocks.append(block)

# Calculate the variance for each block
            variances = [np.var(block) for block in blocks]

# Get the indices of the top 1024 blocks with the highest variance
            top_indices_4096 = np.argsort(variances)[::-1]
            top_indices = top_indices_4096[0:1024]
            print(top_indices[0])
            # top_indices = blocks[top_indices_4096[1024:]]
            # Loop through the image and extract 8x8 blocks
            print(top_indices[0])
            print(top_indices[1023])
            # print(top_indices[4096])
            # def watermark_embedding(P, B, u, size_wm, wt,alpha):
            #     u=0
            #     Q = np.zeros_like(B)
            #     for i in range(2):
            #         if u < size_wm:
            #             if (P[i] - B[i]) > alpha[i] and wt[u] == 0:
            #                 Q[i] = P[i] - alpha[i]
            #                 u = u + 1
            #             elif (B[i] - P[i]) > alpha[i] and wt[u] == 1:
            #                 Q[i] = P[i] + alpha[i]
            #                 u = u + 1
            #             else:
            #                 Q[i] = B[i]
            #     return Q, u
            def watermark_embedding(P, B, u, size_wm, wt, alpha):
                # u = 0
                Q = np.zeros_like(B)
                for i in range(2):
                    if u <= size_wm:
                        if (P - B[i]) > alpha and wt[u] == 0:
                            Q[i] = P - alpha
                            u = u + 1
                        elif (B[i] - P) > alpha and wt[u] == 1:
                            Q[i] = P + alpha
                            u = u + 1
                        else:
                            Q[i] = B[i]
                return Q, u


            for image_block_index in top_indices:
                    image_block = blocks[image_block_index]


                    variances2.append(np.var(image_block))



                    # image_block1=cv2.resize(image_block,(256,256))
                    # cv2.imshow('Image', image_block1)
                    
                    
                    
                    # Apply DWT to the block
                    LL, (LH, HL, HH) = pywt.dwt2(image_block, 'bior1.3')

                    # Apply DCT on LL sub-band
                    dct = cv2.dct(LL)

                    # Select coefficients b(2,3) and b(3,2)
                    b23 = dct[2, 3]
                    b32 = dct[3, 2]

                    
                    # Print the values
                    # print(f"Block {image_block_index}: b23 = {b23}, b32 = {b32}")
                    
                    
                    # Calculate the values required for watermark embedding rules
                    # alpha_value = (HL[2, 2] + LH[2, 2]) / 16.0
                    block_average = np.mean(LL)
                    # alpha_value = (b23 + b32 + 16.0 * block_average)/block_average
                    alpha_value=16.0
                    
                    P = block_average / alpha_value
                    u=0
                    size_wm=32
                    # wt = [0, 1, 0, 1, 0, 1, 0, 1]
                    # wt = np.random.randint(2,size=size_wm)
                    wt = cv2.threshold(watermark_image, 128, 1, cv2.THRESH_BINARY)
                

            # wt = np.random.randint(2, size=size_wm)
                    if u < size_wm:  # Make sure to set size_wm and wt as needed
                        Q, u = watermark_embedding(P,(b23, b32), u, size_wm, wt,alpha_value)
                        # (b23, b32), u = watermark_embedding((b23, b32), (LL[0, 0], LL[0, 1]), u, size_wm, wt)
                        # print(f"Block {image_block_index}: Modified b23 = {b23}, Modified b32 = {b32}")


                    # Reconstruct the watermarked block using inverse DCT
                    LL[2, 3] = b23
                    LL[3, 2] = b32
                    watermarked_block_dct = cv2.idct(LL)
                    # print(f"Block {image_block_index}: Modified b23 = {b23}, Modified b32 = {b32}")


                    # Apply inverse DWT to the watermarked block
                    watermarked_block = pywt.idwt2((watermarked_block_dct, (LH, HL, HH)), 'bior1.3')
                    ll_bands.append(watermarked_block)



                    # for image_block_index in top_indices:
                        # image_block = blocks[image_block_index]


                    # image_block1=cv2.resize(image_block,(256,256))
                    # cv2.imshow('Image', image_block1)


            

            # Assemble the watermarked 8x8 blocks into the original image
           # watermarked_image = np.zeros((height, width), dtype=np.uint8)
            # block_idx = 0

            # for y in range(0, height, 8):
            #     for x in range(0, width, 8):
            #         if block_idx < len(ll_bands):
            #             host_image[y:y+8, x:x+8] = ll_bands[block_idx]
            #             block_idx += 1
             # Replace the top_indices image blocks in the host image with the watermarked blocks
            for i, image_block_index in enumerate(top_indices):
                y = (image_block_index // (width // 8)) * 8
                x = (image_block_index % (width // 8)) * 8
                host_image[y:y+8, x:x+8] = ll_bands[i]
            print(f"Block {image_block_index}: Original Variance = {variances[0]}, Embedded Variance = {variances2[0]}")

            # cv2.imshow("Block After Processing", cv2.resize(ll_bands[0],(256,256)))
            
            # Display the host image with the watermark embedded
            cv2.imshow("Host Image with Watermark", host_image)
            cv2.imshow("Original image", original_image)
            cv2.imwrite("watermarked6.jpg",host_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()