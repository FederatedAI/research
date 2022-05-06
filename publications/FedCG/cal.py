import numpy as np

nums = [[0.797000, 0.778000, 0.783000, 0.778400, 0.790200],
        [0.761447, 0.779348, 0.768823, 0.769438, 0.775430],
        [0.852000, 0.866000, 0.853778, 0.854444, 0.852444],
        [0.965591, 0.972043, 0.962366, 0.965591, 0.970968]]

nums = np.array(nums)
nums = nums.T
       
avg_nums = []
for i in range(4):
    print(np.mean(nums[i]), np.std(nums[i])) 
    avg_nums.append(np.mean(nums[i]))

print(np.mean(avg_nums), np.std(avg_nums))
