# Offline fixtures shared by the test files. Nothing here touches the network.

using DataFrames
using SDMXer

const DEMO_CSV = normpath(joinpath(@__DIR__, "..", "demo", "korea", "inbound_visitors.csv"))

"""
    fixture_schema(; with_codelists=true) -> DataflowSchema

Small tourism-like schema modelled on OECD DF_INBOUND: four dimensions, a time
dimension, one measure, one mandatory and one conditional attribute. With
`with_codelists=true` the dimensions and the unit attribute carry codelist ids
that match `fixture_codelists()`.
"""
function fixture_schema(; with_codelists::Bool=true)
    dim_codelists = with_codelists ? ["CL_AREA", missing, "CL_VISITOR_TYPE", "CL_MEASURE"] : fill(missing, 4)
    attr_codelists = with_codelists ? ["CL_UNIT", missing] : fill(missing, 2)
    dimensions = DataFrame(
        dimension_id = ["REF_AREA", "COUNTERPART_AREA", "VISITOR_TYPE", "MEASURE"],
        position = [1, 2, 3, 4],
        concept_id = ["REF_AREA", "COUNTERPART_AREA", "VISITOR_TYPE", "MEASURE"],
        concept_scheme = fill(missing, 4),
        codelist_id = dim_codelists,
        codelist_agency = fill(missing, 4),
        codelist_version = fill(missing, 4),
        data_type = fill(missing, 4),
        is_time_dimension = fill(false, 4)
    )
    attributes = DataFrame(
        attribute_id = ["UNIT_MEASURE", "OBS_STATUS"],
        assignment_status = ["Mandatory", "Conditional"],
        concept_id = ["UNIT_MEASURE", "OBS_STATUS"],
        concept_scheme = fill(missing, 2),
        codelist_id = attr_codelists,
        codelist_agency = fill(missing, 2),
        codelist_version = fill(missing, 2),
        data_type = fill(missing, 2),
        relationship = ["Dimension", "Observation"]
    )
    measures = DataFrame(
        measure_id = ["OBS_VALUE"],
        concept_id = ["OBS_VALUE"],
        concept_scheme = [missing],
        data_type = ["Double"]
    )
    time_dimension = (
        dimension_id = "TIME_PERIOD", position = 5, concept_id = "TIME_PERIOD",
        concept_scheme = missing, codelist_id = missing, codelist_agency = missing,
        codelist_version = missing, data_type = "ObservationalTimePeriod",
        is_time_dimension = true
    )
    dataflow_info = (
        id = "DF_INBOUND", agency = "TEST", version = "1.0",
        name = "Inbound tourism (test)", description = missing, dsd_id = "DSD_TEST"
    )
    return DataflowSchema(dataflow_info, dimensions, attributes, measures, time_dimension)
end

"""
    fixture_codelists() -> DataFrame

Codelists in the shape returned by `SDMXer.extract_all_codelists`.
"""
function fixture_codelists()
    return DataFrame(
        codelist_id = ["CL_AREA", "CL_AREA", "CL_AREA", "CL_VISITOR_TYPE", "CL_VISITOR_TYPE", "CL_MEASURE", "CL_UNIT"],
        code_id = ["CN", "JP", "US", "OVN", "SDV", "ARR", "PS"],
        name = ["China", "Japan", "United States", "Overnight visitors", "Same-day visitors", "Arrivals", "Persons"],
        parent_code_id = Union{String, Missing}[missing, missing, missing, missing, missing, missing, missing]
    )
end
