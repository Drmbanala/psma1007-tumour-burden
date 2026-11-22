* P01 primary mixed models in Stata/MP (protocol v1.0, section 8).
* Input: analysis_table.csv written by 12_models.py (same folder as the results).
* Usage (batch): stata-mp -b do 13_models_stata.do "<results folder>"
* Written 2026-11-14 before any organ-uptake table existed. Not yet run in Stata.
version 18
clear all
set more off
args resdir
if "`resdir'" != "" cd "`resdir'"
capture log close
log using "stata_models.log", replace text
import delimited using "analysis_table.csv", clear varnames(1) case(preserve) bindquote(strict) maxquotedrows(unlimited)
keep if tracer == "18F" & inrange(uptake_series_min, 45, 120)
encode PatientID, gen(pid)
encode scanner, gen(scan)
tempname H
postfile `H' str12 organ str24 term double(beta se pct lo hi p) long(n_scans n_patients) using "stata_models_results.dta", replace
foreach o in liver spleen kidneys {
    preserve
    capture confirm numeric variable y_`o'
    if _rc destring y_`o', replace force
    keep if y_`o' < . & y_`o' > 0 & x < .
    gen double ly = ln(y_`o')
    bysort pid: egen double xm = mean(x)
    gen double xw = x - xm
    foreach v in uptake_series_min age_years weight_kg {
        quietly summarize `v'
        gen double c_`v' = `v' - r(mean)
    }
    * overall association (aim A)
    mixed ly x c_uptake_series_min i.scan c_age_years c_weight_kg || pid:, reml
    post `H' ("`o'") ("overall (aim A)") (_b[x]) (_se[x]) (100*(exp(_b[x])-1)) (100*(exp(_b[x]-1.96*_se[x])-1)) (100*(exp(_b[x]+1.96*_se[x])-1)) (2*normal(-abs(_b[x]/_se[x]))) (e(N)) (e(N_g)[1,1])
    * within- and between-patient effects (aim B)
    mixed ly xw xm c_uptake_series_min i.scan c_age_years c_weight_kg || pid:, reml
    foreach t in xw xm {
        local lab = cond("`t'" == "xw", "within-patient (aim B)", "between-patient")
        post `H' ("`o'") ("`lab'") (_b[`t']) (_se[`t']) (100*(exp(_b[`t'])-1)) (100*(exp(_b[`t']-1.96*_se[`t'])-1)) (100*(exp(_b[`t']+1.96*_se[`t'])-1)) (2*normal(-abs(_b[`t']/_se[`t']))) (e(N)) (e(N_g)[1,1])
    }
    restore
}
postclose `H'
use "stata_models_results.dta", clear
format beta se %9.5f
format pct lo hi %7.2f
list, noobs sep(3)
export delimited using "stata_models_results.csv", replace
log close
