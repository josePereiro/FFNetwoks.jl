struct TestDatum
    input::Vector{Float64};
    output::Vector{Float64};
end

input(t::TestDatum) = t.input;
output(t::TestDatum) = t.output;
inlength(t::TestDatum) = length(t.input);
outlength(t::TestDatum) = length(t.output);
