def writeList(l,name):
	wptr = open(name,"w")
	wptr.write("%d\n" % len(l))
	for i in l:
		wptr.write("%d\n" % i)
	wptr.close()