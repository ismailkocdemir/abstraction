cd activations/
for file in *
do
  cd ~/dl/thesis/
  python3 main.py -a "$file" > architectures/"$file".txt
done
