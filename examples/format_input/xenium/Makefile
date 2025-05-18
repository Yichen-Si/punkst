# --------- Prepare 10X Xenium data

# ─── Configurable parameters ────
DATADIR      := {DATADIR}
RAW_TRANSCRIPTS := {RAW_TRANSCRIPTS}
RAW_CELLS := {RAW_CELLS}

TRANSCRIPTS  := $(DATADIR)/transcripts.tsv
CELLCOORD    := $(DATADIR)/cell_coordinates.tsv

TAB := $(shell printf '\t')

# ─── Step 0. Prepare‐data ─────────────────────────────────────────────────
.PHONY: all
all: $(TRANSCRIPTS) $(CELLCOORD)

$(TRANSCRIPTS):
	@echo "Extracting Xenium transcripts → $@"
	zcat $(RAW_TRANSCRIPTS) \
	  | cut -d',' -f4-6 | sed 's/"//g' \
	  | awk -F',' -v OFS="$(TAB)" '{{ print $$2, $$3, $$1, "1" }}' \
	  > $@

# ─── cell_coordinates.tsv ← sample all cells
$(CELLCOORD):
	zcat $(RAW_CELLS) \
	  | cut -d',' -f2-3 \
	  | tail -n +2 \
	  | awk -F',' -v OFS="$(TAB)" '{{printf "%.4f\t%.4f\n", $$1,$$2;}}' > $(CELLCOORD)
