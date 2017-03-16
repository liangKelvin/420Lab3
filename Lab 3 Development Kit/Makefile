all:
	gcc datagen.c Lab3IO.c -o datagen
	gcc -fopenmp main.c Lab3IO.c -o main
	gcc serialtester.c Lab3IO.c -o serialtester -lm

clean:
	rm datagen
	rm serialtester
	rm main
	rm data_input

run:
	./datagen
	./main
