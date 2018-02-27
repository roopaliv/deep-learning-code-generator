
import matplotlib.pyplot as plt






if __name__ == "__main__":
	#epochs = [19, 39, 59, 79]
	#cer = [0.85, 0.835, 0.64, 0.01]
	#loss = [2.8365, 2.1246, 0.8684, 0.3215]


	epochs = [19, 39, 59, 79, 99]
	#cer = [0.965, 0.9, 0.83, 0.49, 0.005]
	loss = [2.8568, 1.5418, 0.7047, 0.3517, 0.2008]


	plt.xlabel('Epoch')
	#plt.ylabel('Character Error Rate')
	plt.ylabel('Loss')
	#plt.title('Epoch vs Character Error Rate - Code')
	#plt.title('Epoch vs Character Error Rate - Literature')
	#plt.title('Epoch vs Loss - Literature')
	plt.title('Epoch vs Loss - Code')




	#plt.plot(epochs, cer, 'b-')
	plt.plot(epochs, loss, 'r-')
	#plt.show()
	#plt.savefig('poems_EpochVscer.png')
	#plt.savefig('code_EpochVscer.png')
	#plt.savefig('poems_EpochVsloss.png')
	plt.savefig('code_EpochVsloss.png')
	

