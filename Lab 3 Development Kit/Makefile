all:
	gcc datagen.c Lab3IO.c -o datagen
	gcc -fopenmp test.c Lab3IO.c -o test
	gcc serialtester.c Lab3IO.c -o serialtester -lm

clean:
	rm datagen
	rm serialtester
	rm test

run:
	./datagen
	./test