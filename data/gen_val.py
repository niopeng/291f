f = open("chair_val.txt", "w")
for i in range(1, 187):
    f.write("SS_%03d\n" % i)
f.close()