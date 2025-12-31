# Parsing Errors in Recast Model Validation

This document catalogs the Antimony parsing errors encountered when attempting to validate recast models. These errors prevent the validator from loading the recast .ant files.

## Summary

Of 894 successful recasts, 446 could not be validated due to Antimony parsing errors. The dominant issue (86%) is reserved keyword conflicts.

## Error Categories

### 1. Reserved Keyword Conflicts (259 cases, 58%)

**Problem:** ssys generates Antimony output that uses reserved keywords as variable names.

**Example:**
```antimony
model BIOMD0000000009_recast()

compartment compartment = 4e-12;   # ERROR: "compartment" is a reserved keyword
species DNA in compartment;        # ERROR: "DNA" conflicts with built-in
```

**Error message:**
```
syntax error, unexpected "'compartment'", expecting '$' or '.' or element name
syntax error, unexpected "'DNA'", expecting '$' or '.' or element name
```

**Affected reserved words:**
- `compartment` - Very common in BioModels
- `DNA`, `RNA` - Common species names
- `species`, `model`, `function`, `unit` - Antimony keywords

**Fix approach:**
- In `ssys_to_antimony()`, detect reserved keyword conflicts
- Rename conflicting variables with suffix (e.g., `compartment_var`)
- Update all references in equations

### 2. Reserved Function Name Conflicts (~15 cases, 3%)

**Problem:** Variables named after built-in functions cause parsing failures.

**Example:**
```antimony
exp = 1.5;    # ERROR: "exp" is a built-in function
log = 2.0;    # ERROR: "log" is a built-in function
```

**Error message:**
```
syntax error, unexpected name of an existing function
```

**Affected function names:**
- `exp`, `log`, `ln`, `log10`
- `sin`, `cos`, `tan`
- `sqrt`, `pow`, `abs`

**Fix approach:**
- Same as #1: rename conflicting variables

### 3. Unit Definition Invalid (~15 cases, 3%)

**Problem:** Some models have unit definitions with scientific notation that Antimony doesn't accept.

**Error message:**
```
Unable to set a unit definition using the formula '2e-14'. Only multiplication, 
division, and raising a value to a numerical power are allowed.
```

**Fix approach:**
- Strip or convert problematic unit definitions
- This may be an upstream issue from the SBML source

### 4. Assignment + Rate Rule Conflict (~8 cases, 2%)

**Problem:** A variable has both an assignment rule and a rate rule (ODE), which is invalid.

**Example:**
```antimony
Pt := some_expression;        # Assignment rule
Pt' = rate_expression;        # Rate rule - CONFLICT!
```

**Error message:**
```
The variable 'Pt' is associated with an assignment rule, and may not additionally have a rate rule.
```

**Root cause:**
- ssys may be generating rate rules for variables that should be assignment rules
- Or the original model has inconsistent rule types

**Fix approach:**
- During recasting, track which variables are derived (assignment) vs. dynamic (rate)
- Ensure recast output uses consistent rule types

### 5. Undefined Function References (~4 cases, <1%)

**Problem:** References to custom functions that aren't defined in the recast output.

**Example:**
```antimony
v1 = Constant_flux__irreversible(k1);  # ERROR: function not defined
```

**Error message:**
```
'Constant_flux__irreversible' was used as a function, but no such function was defined.
```

**Root cause:**
- Custom function definitions from SBML not being preserved in recast output

**Fix approach:**
- Preserve function definitions from original model
- Or inline the function body during recasting

## Priority for Fixes

1. **Reserved keyword conflicts (259 cases)** - Single fix, high impact
2. **Reserved function name conflicts (~15 cases)** - Same fix as #1
3. **Assignment/rate rule conflicts (~8 cases)** - Requires recasting logic review
4. **Undefined functions (~4 cases)** - Requires function preservation
5. **Unit definitions (~15 cases)** - May require stripping units

## Testing Strategy

After implementing fixes:
```bash
# Re-run validation on affected models
python 3b_validate_batch.py --numerical-only --timeout 60 --resume

# Check for remaining errors
grep -l "Antimony parsing error" results/validation/*.json | wc -l
```

## Related Files

- `src/ssys/recaster.py` - Main recasting logic
- `ssys_to_antimony()` function generates output
- `biomodels_batch/3b_validate_batch.py` - Validation script
- `src/ssys/validator.py` - Validation implementation
