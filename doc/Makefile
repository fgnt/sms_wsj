default: images/example_id.svg

# Does not work for this image
#images/example_id.svg: images/example_id.pdf
#	inkscape -l $@ $^

images/example_id.svg: images/example_id.tex
	cd images && pdflatex -shell-escape example_id.tex

