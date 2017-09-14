function seedids=readSeed(filename)
s = load(filename,'-ascii');
n = s(1);
seedids = zeros(n,1);
for i = 1:n
    seedids(i) = s(i + 1);
end

