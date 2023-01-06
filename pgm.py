import numpy as np

matrix=np.array([[1,2,3],[6,3,9],[2,6,9]])
print(matrix)

a,b,c=np.linalg.svd(matrix)
print("Decomposed matrix are  :",a)
print(b)
print(c)

print(a@ np.diag(b) @c)









