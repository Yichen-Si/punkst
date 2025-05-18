# ─── Configurable parameters (these get filled in by your Python script) ────
GITPATH      := {GITPATH}
DATADIR      := {DATADIR}
TMPDIR       := {TMPDIR}
TRANSCRIPTS  := {TRANSCRIPTS}
ICOL_X       := {ICOL_X}
ICOL_Y       := {ICOL_Y}
ICOL_FEATURE := {ICOL_FEATURE}
ICOL_COUNT   := {ICOL_COUNT}
SKIP         := {SKIP}
EXCLUDE_FEATURE_REGEX := {EXCLUDE_FEATURE_REGEX}

THREADS      := {THREADS}
SEED         := {SEED}
TILESIZE     := {TILESIZE}
HEXGRIDS     := {HEXGRIDS}
TOPICS       := {TOPICS}
PIXHEX       := {PIXHEX}
NMOVE        := {NMOVE}
RES          := {RES}
SCALE        := {SCALE}

PUNKST       := $(GITPATH)/bin/punkst
TILETSV      := $(DATADIR)/transcripts.tiled.tsv
TILEINDEX    := $(DATADIR)/transcripts.tiled.index
COORD_RANGE  := $(DATADIR)/transcripts.tiled.coord_range.tsv
FEATURES     := $(DATADIR)/transcripts.tiled.features.tsv

# ─── Compute the full list of final report files ──────────────────────────
REPORTS := \
  $(foreach H,$(HEXGRIDS),\
    $(foreach K,$(TOPICS),\
      $(DATADIR)/k$(K)_hex_$(H).$(PIXHEX)_$(NMOVE)_$(RES).factor.info.html\
	  $(DATADIR)/k$(K)_hex_$(H).$(PIXHEX)_$(NMOVE)_$(RES).pixel.png\
    )\
  )

.PHONY: all clean clean_all
all: $(REPORTS)

# ─── Step 1: bag pixels → tiles.tsv + tiles.index ─────────────────────
$(DATADIR)/prepare.stamp: $(TRANSCRIPTS)
	@echo "[1/6] pts2tiles → $@"
	 $(PUNKST) pts2tiles \
	  --in-tsv $(TRANSCRIPTS) \
	  --icol-x $(ICOL_X) --icol-y $(ICOL_Y) \
	  --icol-feature $(ICOL_FEATURE) --icol-int $(ICOL_COUNT) --skip $(SKIP) \
	  --tile-size $(TILESIZE) \
	  --tile-buffer 1000 \
	  --out-prefix $(DATADIR)/transcripts.tiled \
	  --temp-dir $(TMPDIR) --threads $(THREADS)
	touch $(DATADIR)/prepare.stamp

# ─── Step 2: for each H in HEXGRIDS → hex_H_randomized.txt ────────────
define HEX_RULE
$(DATADIR)/hex_$(1)_randomized.txt: \
    $(DATADIR)/prepare.stamp
	@echo "[2/6] tiles2hex (hex=$(1)) → $(DATADIR)/hex_$(1)_randomized.txt"
	 $(PUNKST) tiles2hex \
	  --in-tsv $(TILETSV) \
	  --in-index $(TILEINDEX) \
	  --feature-dict $(FEATURES) \
	  --icol-x $(ICOL_X) --icol-y $(ICOL_Y) \
	  --icol-feature $(ICOL_FEATURE) \
	  --icol-int $(ICOL_COUNT) \
	  --hex-grid-dist $(1) \
	  --out $(DATADIR)/hex_$(1).txt \
	  --temp-dir $(TMPDIR) --threads $(THREADS)
	sort -k1,1 --parallel $(THREADS) -S 2G \
	  $(DATADIR)/hex_$(1).txt > $(DATADIR)/hex_$(1)_randomized.txt
	rm $(DATADIR)/hex_$(1).txt
endef
$(foreach H,$(HEXGRIDS),$(eval $(call HEX_RULE,$H)))

# ─── Step 3: for each (K,H) → model.tsv + color.rgb.tsv ──────────────
define LDA_MODEL_RULE
$(DATADIR)/k$(2)_hex_$(1).model.stamp: \
    $(DATADIR)/hex_$(1)_randomized.txt
	@echo "[3/6] lda4hex (K=$(2), hex=$(1)) → $(DATADIR)/k$(2)_hex_$(1).model.tsv"
	 $(PUNKST) lda4hex \
	  --in-data $(DATADIR)/hex_$(1)_randomized.txt \
	  --in-meta $(DATADIR)/hex_$(1).json \
	  --features $(FEATURES) \
	  --n-topics $(2) \
	  --out-prefix $(DATADIR)/k$(2)_hex_$(1) --transform \
	  --min-count-per-feature 50 --min-count-train 50 --minibatch-size 512 \
	  --exclude-feature-regex $(EXCLUDE_FEATURE_REGEX) \
	  --threads $(THREADS) --seed $(SEED) --n-epochs 1
	touch $(DATADIR)/k$(2)_hex_$(1).model.stamp
endef
$(foreach H,$(HEXGRIDS),\
  $(foreach K,$(TOPICS),\
    $(eval $(call LDA_MODEL_RULE,$H,$K))\
  )\
)

define COLOR_RULE
$(DATADIR)/k$(2)_hex_$(1).color.rgb.tsv: \
    $(DATADIR)/k$(2)_hex_$(1).model.stamp
	@echo "[3/6] color_helper (K=$(2), hex=$(1)) → $$@"
	python $(GITPATH)/ext/py/color_helper.py \
	  --input $(DATADIR)/k$(2)_hex_$(1).results.tsv \
	  --output $(DATADIR)/k$(2)_hex_$(1).color
endef
$(foreach H,$(HEXGRIDS),\
  $(foreach K,$(TOPICS),\
    $(eval $(call COLOR_RULE,$H,$K)) \
  )\
)

# ─── Step 4: pixel-decode → .tsv + .pseudobulk.tsv ────────────────────
define PIXEL_RULE
$(DATADIR)/k$(2)_hex_$(1).$(PIXHEX)_$(NMOVE)_$(RES).decode.stamp: \
	$(DATADIR)/prepare.stamp \
    $(DATADIR)/k$(2)_hex_$(1).model.stamp
	@echo "[4/6] pixel-decode (K=$(2), hex=$(1)) → $$*.tsv"
	 $(PUNKST) pixel-decode \
	  --model $(DATADIR)/k$(2)_hex_$(1).model.tsv \
	  --in-tsv $(TILETSV) \
	  --in-index $(TILEINDEX) \
	  --out-pref $(DATADIR)/k$(2)_hex_$(1).$(PIXHEX)_$(NMOVE)_$(RES) \
	  --icol-x $(ICOL_X) --icol-y $(ICOL_Y) \
	  --icol-feature $(ICOL_FEATURE) --icol-val $(ICOL_COUNT) \
	  --hex-grid-dist $(PIXHEX) --n-moves $(NMOVE) \
	  --min-init-count 20 --pixel-res $(RES) \
	  --temp-dir $(TMPDIR) --threads $(THREADS) --seed $(SEED) \
	  --output-original --output-coord-digits 2 --output-prob-digits 4
	touch $(DATADIR)/k$(2)_hex_$(1).$(PIXHEX)_$(NMOVE)_$(RES).decode.stamp
endef
$(foreach H,$(HEXGRIDS),\
  $(foreach K,$(TOPICS),\
    $(eval $(call PIXEL_RULE,$H,$K))\
  )\
)

# ─── Step 5: draw-pixel-factors → png ───────────────────────────
define DRAW_RULE
$(DATADIR)/k$(2)_hex_$(1).$(PIXHEX)_$(NMOVE)_$(RES).pixel.png: \
    $(DATADIR)/k$(2)_hex_$(1).$(PIXHEX)_$(NMOVE)_$(RES).decode.stamp \
    $(DATADIR)/k$(2)_hex_$(1).color.rgb.tsv
	@echo "[5/6] draw-pixel-factors (K=$(2), hex=$(1)) → $$@"
	 $(PUNKST) draw-pixel-factors \
	  --in-tsv $(DATADIR)/k$(2)_hex_$(1).$(PIXHEX)_$(NMOVE)_$(RES).tsv \
	  --in-color $(DATADIR)/k$(2)_hex_$(1).color.rgb.tsv \
	  --out $$@ \
	  --range $(COORD_RANGE) --scale $(SCALE)
endef
$(foreach H,$(HEXGRIDS),\
  $(foreach K,$(TOPICS),\
    $(eval $(call DRAW_RULE,$H,$K))\
  )\
)

# ─── Step 6: de_bulk + factor_report → final .report ───────────────
define DE_RULE
$(DATADIR)/k$(2)_hex_$(1).$(PIXHEX)_$(NMOVE)_$(RES).de_bulk.tsv: \
    $(DATADIR)/k$(2)_hex_$(1).$(PIXHEX)_$(NMOVE)_$(RES).decode.stamp
	@echo "[6/6] de_bulk (K=$(2), hex=$(1)) → $$@"
	python $(GITPATH)/ext/py/de_bulk.py \
	  --input $(DATADIR)/k$(2)_hex_$(1).$(PIXHEX)_$(NMOVE)_$(RES).pseudobulk.tsv \
	  --output $(DATADIR)/k$(2)_hex_$(1).$(PIXHEX)_$(NMOVE)_$(RES).de_bulk.tsv \
	  --feature_label Feature --thread $(THREADS)
endef
$(foreach H,$(HEXGRIDS),\
  $(foreach K,$(TOPICS),\
    $(eval $(call DE_RULE,$H,$K))\
  )\
)

define REPORT_RULE
$(DATADIR)/k$(2)_hex_$(1).$(PIXHEX)_$(NMOVE)_$(RES).factor.info.html: \
    $(DATADIR)/k$(2)_hex_$(1).$(PIXHEX)_$(NMOVE)_$(RES).decode.stamp \
	$(DATADIR)/k$(2)_hex_$(1).$(PIXHEX)_$(NMOVE)_$(RES).de_bulk.tsv \
    $(DATADIR)/k$(2)_hex_$(1).color.rgb.tsv
	@echo "[6/6] factor_report (K=$(2), hex=$(1)) → $$@"
	python $(GITPATH)/ext/py/factor_report.py \
	  --de $(DATADIR)/k$(2)_hex_$(1).$(PIXHEX)_$(NMOVE)_$(RES).de_bulk.tsv \
	  --pseudobulk $(DATADIR)/k$(2)_hex_$(1).$(PIXHEX)_$(NMOVE)_$(RES).pseudobulk.tsv \
	  --feature_label Feature \
	  --color_table $(DATADIR)/k$(2)_hex_$(1).color.rgb.tsv \
	  --output_pref $(DATADIR)/k$(2)_hex_$(1).$(PIXHEX)_$(NMOVE)_$(RES)
endef
$(foreach H,$(HEXGRIDS),\
  $(foreach K,$(TOPICS),\
    $(eval $(call REPORT_RULE,$H,$K))\
  )\
)

# ─── Clean up parameter specific files ──────────────────────────────────
clean:
	rm -rf $(TMPDIR) \
	       $(DATADIR)/hex_* \
	       $(DATADIR)/k*_hex_*
clean_all:
	rm -rf $(TMPDIR) \
		   $(DATADIR)/transcripts.tiled.* \
	       $(DATADIR)/hex_* \
	       $(DATADIR)/k*_hex_*
