def mergeKLists(lists):
	#Idea: Pop the largest leading term from among the lists, and add it to a new list
	#Delete empty lists from lists
	merged = []
	while lists != []:
		print(lists)
		print(merged)
		while (lists != []) and (lists[0] == []):
			lists.pop(0)
		leastsofar = lists[0][0]
		best = 0
		k = 0
		while k < len(lists):
			while (k < len(lists)) and (lists[k] == []):
				lists.pop(k)
			#In case a person submitted empty lists
			if k < len(lists):
				if leastsofar > lists[k][0]:
						leastsofar = lists[k][0]
						best = k
			k += 1
		merged.append(leastsofar)
		lists[best].pop(0)
		if lists[best] == []:
			lists.pop(best)

	return merged