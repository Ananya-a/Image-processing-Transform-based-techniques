import cv2
import numpy as np
import pywt

# Load the host image
host_image = cv2.imread('lenna.png', cv2.IMREAD_GRAYSCALE)
original_image = cv2.imread('lenna.png', cv2.IMREAD_GRAYSCALE)
watermark_image = cv2.imread('Binary2.png', cv2.IMREAD_GRAYSCALE)
# _,wt = cv2.threshold(watermark_image, 128, 1, cv2.THRESH_BINARY)
# wt = wt.flatten() 
# print(wt)
# Print the first 10 bits
# print(wt[:-1])

# Alternatively, you can loop through the array and print each bit
# for bit in wt:
    # print(bit, end=' ')

if watermark_image is None:
        print("Error: Unable to load the watermark image.")
else:
         # Threshold the watermark image to convert it into binary data
        _,wt = cv2.threshold(watermark_image, 128, 1, cv2.THRESH_BINARY)
        # print(wt)

        # Get the dimensions of the host image
        height, width = host_image.shape

        # Check if the dimensions are divisible by 8 (for simplicity)
        if height % 8 != 0 or width % 8 != 0:
            print("Image dimensions are not divisible by 8. Adjust the image size.")
        else:
            # Initialize an empty list to store the LL bands after DWT
            ll_bands = []
            ll_subbands=[]
            dct_subands=[]
            idct_subbands=[]
            embedded_block_coordinates = []
            blocks=[]
            count=0
            count1=0
            for i in range(0, 512, 8):
                for j in range(0, 512, 8):
                    block = host_image[i:i+8, j:j+8]
                    blocks.append(block)

# Calculate the variance for each block
            variances = [np.var(block) for block in blocks]

# Get the indices of the top 1024 blocks with the highest variance
            top_indices_4096 = np.argsort(variances)[::-1]
            top_indices = top_indices_4096[0:1024]
            print(top_indices)
            # cv2.imshow('Image0', cv2.resize(blocks[top_indices[1023]],(256,256)))

            
            # print(top_indices[4096])
            
            def watermark_embedding(P, B, u, size_wm, wt, alpha):
                Q = np.zeros_like(B)
                u=0
                for i in range(2):
                    if u < size_wm:
                        if (P - B[i]) > alpha and np.any(wt[u]) == 0:
                            Q[i] = P - alpha
                            u += 1
                        elif (B[i] - P) > alpha and np.any(wt[u]) == 1:
                            Q[i] = P + alpha 
                            u += 1
                        else:
                            Q[i] = B[i]
                    else:
                        Q[i] = B[i]
                # print(u)
                return Q, u
            # Inside the watermark_embedding function, replace the loop with this code:
            

            for image_block_index in top_indices:
            # for i, image_block_index in enumerate(top_indices):

                    image_block = blocks[image_block_index]

                    # Apply DWT to the block
                    LL, (LH, HL, HH) = pywt.dwt2(image_block, 'bior1.3')
                    # cv2.imshow('Imagedwt', cv2.resize(LL[0],(256,256)))
                    ll_subbands.append(LL)
                    # Access LL sub-band for block 100 and display it
                    # print(len(ll_subbands))
                    # cv2.imshow('LL Sub-band for Block 100', cv2.resize(LL[0], (256, 256)))
                    # else:
                        # print("Block 100 not available.")
                    # cv2.imshow('Imagedwt', cv2.resize(ll_subbands[100],(256,256)))


                    # Apply DCT on LL sub-band
                    dct = cv2.dct(LL)
                    dct_subands.append(LL)
                    # dct2=cv2.dct(dct1)
                    # cv2.imshow('Imagedct', cv2.resize(dct[0],(256,256)))


                    # Select coefficients b(2,3) and b(3,2)
                    b23 = dct[2, 3]
                    b32 = dct[3, 2]

                    # Calculate the values required for watermark embedding rules
                    block_average = np.mean(LL)
                    alpha_value = (b23 + b32 + 16.0 * block_average)/block_average
                    #alpha_value=0.5
                    P = block_average / alpha_value
                    #print(P)
                    size_wm=1024 # len(wt)
                    u=0
                    # wt = [1, 1, 0, 0, 0, 0, 0, 1]
                    # _,wt = cv2.threshold(watermark_image, 128, 1, cv2.THRESH_BINARY)

           
                    # if u < size_wm:  # Make sure to set size_wm and wt as needed
                    Q, u = watermark_embedding(P,(b23, b32),u, size_wm, wt, alpha_value)
                    # embed_cnt+=1
            

                    # Reconstruct the watermarked block using inverse DCT
                    dct[2, 3] = Q[0]
                    dct[3, 2] = Q[1]
                    

                    # if b32!=0 or b32!=0:
                    #      count1=count1+1
                    # if b23-Q[0]!=0 or b32-Q[1]!=0:
                    #      count=count+1
                    watermarked_block_dct = cv2.idct(dct)
                    idct_subbands.append(watermarked_block_dct)
                    # watermarked_block_dct2 = cv2.idct(dct2)

                    # cv2.imshow('Imageidct', cv2.resize(watermarked_block_dct[0],(256,256)))


                    # Apply inverse DWT to the watermarked block
                    watermarked_block = pywt.idwt2((watermarked_block_dct, (LH, HL, HH)), 'bior1.3')
                    ll_bands.append(watermarked_block)

                    if b32!=0 or b32!=0:
                         count1=count1+1
                    if b23-Q[0]!=0 or b32-Q[1]!=0:
                         count=count+1
                    


            # cv2.imshow('Imagedwt', cv2.resize(ll_subbands[1023], (256, 256)))
            # cv2.imshow('Imagedct', cv2.resize(dct_subands[1023],(256,256)))
            # cv2.imshow('Imageidct', cv2.resize(idct_subbands[1023],(256,256)))
            # cv2.imshow('Imageidwt', cv2.resize(ll_bands[1023],(256,256)))
            # cv2.imshow('Image2', cv2.resize(blocks[top_indices[1023]],(256,256)))


            # else:
                # print("Block 100 not available.")
            print(count1,count)
            for i,image_block_index in enumerate(top_indices):
                y = (image_block_index // (width // 8)) * 8
                x = (image_block_index % (width // 8)) * 8
                host_image[y:y+8, x:x+8] = ll_bands[i]
                cv2.imshow('Image', cv2.resize(blocks[top_indices[1023]],(256,256)))
                embedded_block_coordinates.append((x, y))
            print(embedded_block_coordinates[1023])



            # print(embed_cnt)
            # image_block1=cv2.resize(image_block,(256,256))
            
            # Display the host image with the watermark embedded
            cv2.imshow("Host Image with Watermark", host_image)
            cv2.imshow("Original image", original_image)
            cv2.imwrite("watermarked8.jpg",host_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()