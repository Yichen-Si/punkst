#ifndef SCRIBBLE_H
#define SCRIBBLE_H

#include "utils.h"
#include "qgenlib/seq_utils.h"
#include <regex>
#include <sstream>
#include <iostream>
#include <fstream>

class ScribbleConfig {
    public:

    std::string upstreamPrimer, staticBarcode;
    std::string downstreamFixed, nickingFixed, captureSequence;
    std::map<std::string, std::string> protospacerFull, protospacer, mutableBarcode, spacerTail;
    std::map<std::string, std::vector<std::string> > validPair;
    std::vector<std::string> protospacerIds;
    int32_t npair;
    int32_t ntEditDelete = 3;
    int32_t ntSpacer, ntBarcode;
    uint32_t variableSpacer_min = 0, variableSpacer_max = 4;
    int32_t upPrimLen, statBcLen;
    int32_t downFixLen, nickFixLen, capSeqLen, minTail;
    ScribbleConfig(std::string& config) {
        ParseConfig(config);
    }

    void ParseConfig(std::string& config) {
        std::ifstream ifs(config.c_str(), std::ifstream::in);
        std::string line;
        std::string currentCategory;
        if (!ifs.is_open()) {
            error("Unable to open config file %s", config.c_str());
        }
        while (std::getline(ifs, line)) {
            line = strip_str(line);
            if (line[0] == '@') {
                currentCategory = line;
            } else if (line[0] == '>') {
                std::string key = line.substr(1); // Remove '>' to get the key
                if (std::getline(ifs, line)) { // Read the next line for the value
                    if (currentCategory == "@ProtospacerFull") {
                        protospacerFull[key] = line;
                    } else if (currentCategory == "@Protospacer") {
                        protospacer[key] = line;
                    } else if (currentCategory == "@Barcode") {
                        mutableBarcode[key] = line;
                    } else if (currentCategory == "@ValidPair") {
                        split(validPair[key], ",", line, UINT_MAX, true, true, true);
                    }
                }
            } else { // Handle the single-line values
                if (currentCategory == "@UpstreamPrimer") {
                    upstreamPrimer = line;
                } else if (currentCategory == "@StaticBarcode") {
                    staticBarcode = line;
                } else if (currentCategory == "@DownstreamSequence") {
                    downstreamFixed = line;
                } else if (currentCategory == "@NickingSite") {
                    nickingFixed = line;
                } else if (currentCategory == "@CaptureSequence") {
                    captureSequence = line;
                } else if (currentCategory == "@EditDeletion") {
                    if (!str2int32(line, ntEditDelete)) {
                        error("Invalid @EditDeletion %s", line.c_str());
                    }
                } else if (currentCategory == "@VariableSpacer") {
                    std::vector<std::string> tokens;
                    split(tokens, ",", line, 2, true, true, true);
                    bool flag = true;
                    if (tokens.size() == 2) {
                        if (str2uint32(tokens[0], variableSpacer_min) &&
                            str2uint32(tokens[1], variableSpacer_max)) {
                            if (variableSpacer_max < variableSpacer_min) {
                                variableSpacer_max += variableSpacer_min;
                                variableSpacer_min = variableSpacer_max - variableSpacer_min;
                                variableSpacer_max -= variableSpacer_min;
                            }
                            flag = false;
                        }
                    }
                    if (flag) {
                        error("Invalid @VariableSpacer %s", line.c_str());
                    }
                }
            }
        }
        ifs.close();
        if (protospacerFull.size() > protospacer.size()) {
            if (protospacer.size() == 0) {
                if (ntEditDelete == 0) {
                    protospacer = protospacerFull;
                } else {
                    for (auto& it : protospacerFull) {
                        protospacer[it.first] = it.second.substr(0, it.second.size() - ntEditDelete);
                    }
                }
            } else {
                error("More ProtospacerFull than Protospacer are specified");
            }
        }
        // check if the validPair contains only valid pairs of protospacer and mutable barcodes
        for (auto& it : validPair) {
            if(protospacer.find(it.first) == protospacer.end()) {
                error("@ValidPair contains undefined protospacer %s.", it.first.c_str());
            }
            for (auto& v : it.second) {
                if (mutableBarcode.find(v) == mutableBarcode.end()) {
                    error("@ValidPair contains undefined barcode %s.", v.c_str());
                }
            }
        }
        auto it = protospacer.begin();
        while (it != protospacer.end()) { // delete spacers without valid barcide
            if (validPair.find(it->first) == validPair.end()) {
                notice("ValidPair is not defined for protospacer %s", it->first.c_str());
                it = protospacer.erase(it);
            } else {
                protospacerIds.push_back(it->first);
                ++it;
            }
        }
        if (protospacer.size() == 0) {
            error("No valid protospacer is defined in the config file");
        }
        if (mutableBarcode.size() == 0) {
            error("No valid barcode is defined in the config file");
        }
        ntSpacer = protospacer.begin()->second.size();
        ntBarcode = mutableBarcode.begin()->second.size();
        npair = validPair.size();
        if (ntEditDelete > 0) {
            for (auto& it : protospacerFull) {
                spacerTail[it.first] = it.second.substr(ntSpacer, ntEditDelete);
            }
        }
        if (upstreamPrimer.empty()) {
            notice("@UpstreamPrimer is not defined in the config file");
        }
        if (staticBarcode.empty()) {
            notice("@StaticBarcode is not defined in the config file");
        }
        if (downstreamFixed.empty()) {
            notice("@DownstreamFixed is not defined in the config file");
        }
        if (nickingFixed.empty()) {
            notice("@NickingFixed is not defined in the config file");
        }
        upPrimLen = upstreamPrimer.size();
        statBcLen = staticBarcode.size();
        downFixLen = downstreamFixed.size();
        nickFixLen = nickingFixed.size();
        capSeqLen = captureSequence.size();
        minTail = std::min({downFixLen, nickFixLen, capSeqLen});
    }

    /** find the substring in seq that matches the iupac pattern stored in staticBarcode
    Try start positions within min_offset to max_offset in seq (0-based)
    return the minimum numebr of mismatches
    */
    int32_t FindStatisBarcodeFromFront(char* seq, int32_t l, int32_t& offset, int32_t min_offset=0, int32_t max_offset=0) {
        if (min_offset + statBcLen > l) {
            notice("Input sequence is too short");
            return -1;
        }
        if (max_offset + statBcLen > l) {
            notice("Offset range is too large or input sequence is too short");
            max_offset = l - statBcLen;
        }
        int32_t min_score = statBcLen + 1;
        offset = min_offset;
        for (int32_t i = min_offset; i <= max_offset; ++i) {
            int32_t score = seq_iupac_mismatch(seq+i, staticBarcode.c_str(), statBcLen);
            if (score < min_score) {
                min_score = score;
                offset = i;
                if (min_score == 0) {
                    return 0;
                }
            }
        }
        return min_score;
    }
};

#endif
