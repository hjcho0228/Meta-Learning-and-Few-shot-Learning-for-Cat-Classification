import kagglehub

# Download latest version
path = kagglehub.dataset_download("priyerana/imagenet-10k")

print("Path to dataset files:", path)