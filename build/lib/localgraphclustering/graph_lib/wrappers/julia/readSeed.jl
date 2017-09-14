function readSeed(filename::AbstractString)
    (rows,header) = readdlm(filename;header=true)
    A =convert(Array{Int64,1},rows[1:parse(Int,header[1]),1])+1
    return A
end
