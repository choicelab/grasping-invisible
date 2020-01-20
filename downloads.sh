echo "Downloading model weights..."
fileid="1rHSX0XhmBcB3G2bMfjJGXro1uYYAR5XO"
filename="pre-trained.zip"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}
rm cookie
unzip -q pre-trained.zip
rm pre-trained.zip
mv pre-trained/resnet50.pth.tar ./light_weight_refinenet/resnet50.pth.tar
echo "Done."
