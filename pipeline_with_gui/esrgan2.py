import requests
import base64

def upscale_image(image_path, output_path="output_image.png"):
    """
    Takes an image file path, sends it to the ESRGAN API for upscaling, 
    and saves the returned image to the specified output path.

    :param image_path: Path to the input image file.
    :param output_path: Path to save the output image (default: output_image.png).
    :return: Path to the saved output image if successful, else None.
    """
    api_key = "SG_5bd8f011dfcfbede"
    url = "https://api.segmind.com/v1/esrgan"

    # Convert image to base64
    with open(image_path, 'rb') as f:
        image_base64 = base64.b64encode(f.read()).decode('utf-8')

    # Request payload
    data = {
        "image": image_base64,
        "scale": 2
    }

    headers = {'x-api-key': api_key}

    # Send request
    response = requests.post(url, json=data, headers=headers)

    # Save the response image if successful
    if response.status_code == 200:
        with open(output_path, "wb") as f:
            f.write(response.content)
        return output_path
    else:
        print("Failed to get image:", response.text)
        return None

# # Example usage
# output = upscale_image("input.jpg")
# if output:
#     print(f"Upscaled image saved at: {output}")