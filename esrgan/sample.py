import os
import replicate
os.environ["REPLICATE_API_TOKEN"] = "r8_LFtzYbACT5kmQSLszeJX4mujxG4Mqed1BLeiD"

print(replicate.Client().models.list())
image = open("C:/Users/Nehal/Desktop/Ancient_text/improved_output.png", "rb")
input = {
    "image": image
}

output = replicate.run(
    "xinntao/esrgan:c263265e04b16fda1046d1828997fc27b46610647a3348df1c72fbffbdbac912",
    input=input
)
with open("output", "wb") as file:
    file.write(output.read())