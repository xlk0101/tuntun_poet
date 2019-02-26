
test = 138.17231900000002
m, s = divmod(test, 60)
h, m = divmod(m, 60)
print ("%02d:%02d:%02d" % (h, m, s))
