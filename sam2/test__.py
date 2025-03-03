import torch

print(f"GPU disponible : {torch.cuda.is_available()}")
print(f"Nombre de GPU : {torch.cuda.device_count()}")
print(f"Nom du GPU utilisé : {torch.cuda.get_device_name(0)}")
print(f"GPU actuel : {torch.cuda.current_device()}")
