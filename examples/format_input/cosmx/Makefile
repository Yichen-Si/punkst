# --------- Prepare Vizgen MERSCOPE data

# ─── Configurable parameters ────
RAW_TX   := {RAW_TX}
RAW_META := {RAW_META}
MICRONS_PER_PIXEL := {MICRONS_PER_PIXEL}
DATADIR      := {DATADIR}

TRANSCRIPTS  := $(DATADIR)/transcripts.tsv
CELLCOORD    := $(DATADIR)/cell_coordinates.tsv

TAB := $(shell printf '\t')

# ─── Step 0. Prepare‐data ─────────────────────────────────────────────────
.PHONY: all
all: $(CELLCOORD) $(TRANSCRIPTS)

$(CELLCOORD):
	cut -d',' -f 7-8 $(RAW_META) | tail -n +2 | awk -F',' -v OFS="$(TAB)" \
		-v mu=$(MICRONS_PER_PIXEL) \
	    '{{printf "%.2f\t%.2f\n", mu * $$1, mu * $$2;}}' > $(CELLCOORD)

$(TRANSCRIPTS):
	@echo "Extracting transcripts..."
	  awk -F',' -v mu=$(MICRONS_PER_PIXEL) '\
	  NR==1{{gsub(/"/, "", $$0); print $$3, $$4, $$8, "count", $$7, $$9 }}\
	  NR>1{{gsub(/"/, "", $$8); gsub(/"/, "", $$9); printf "%.2f\t%.2f\t%s\t%d\t%d\t%s\n", mu*$$3, mu*$$4, $$8, 1, $$7, $$9 }} ' $(RAW_TX) > $@
