# --------- Prepare Vizgen MERSCOPE data

# ─── Configurable parameters ────
RAWDIR	     := {RAWDIR}
COMPRESSED   := {COMPRESSED}
DATADIR      := {DATADIR}

TRANSCRIPTS  := $(DATADIR)/transcripts.tsv
CELLCOORD    := $(DATADIR)/cell_coordinates.tsv

TAB := $(shell printf '\t')

ifeq ($(COMPRESSED), 1)
	RAW_METADATA = $(RAWDIR)/cell_metadata.csv.gz
	CAT_CMD = zcat
else
	RAW_METADATA = $(RAWDIR)/cell_metadata.csv
	CAT_CMD = cat
endif

# ─── Step 0. Prepare‐data ─────────────────────────────────────────────────
.PHONY: all
all: $(CELLCOORD) $(TRANSCRIPTS)

$(CELLCOORD):
	$(CAT_CMD) $(RAW_METADATA) | cut -d',' -f 4-9 | tail -n +2 | awk -F',' -v OFS="\t" '{{ print $$1, $$2; }}' > $(CELLCOORD)

$(TRANSCRIPTS):
ifeq ($(COMPRESSED),1)
	@echo "Extracting transcripts from compressed data..."
	zcat $(RAWDIR)/detected_transcripts.csv.gz \
	  | cut -d',' -f2-5,9 \
	  | sed \
	      -e '0,/barcode/{{s/barcode/#barcode/}}' \
	      -e 's/,/$(TAB)/g' \
	      -e 's/$$/$(TAB)1/' \
	      -e '0,/barcode/{{s/$(TAB)1$$/$(TAB)count/}}' \
	    > $@
else
	@echo "Extracting transcripts from uncompressed data..."
	cut -d',' -f2-5,9 $(RAWDIR)/detected_transcripts.csv \
	  | sed \
	      -e '0,/barcode/{{s/barcode/#barcode/}}' \
	      -e 's/,/$(TAB)/g' \
	      -e 's/$$/$(TAB)1/' \
	      -e '0,/barcode/{{s/$(TAB)1$$/$(TAB)count/}}' \
	    > $@
endif
