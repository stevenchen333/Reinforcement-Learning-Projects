g = 0
rewards = []
r = 1

for i in range(10):
    g+= r
    print(g)
    rewards.append(g)

print(rewards)

print(sum(rewards))
print(sum(rewards)/len(rewards))