from bisect import bisect_right
from ipaddress import ip_address, ip_network

def compile_ranges(cidrs):
    groups = {4: [], 6: []}
    for cidr in cidrs:
        net = ip_network(cidr, strict=True)
        groups[net.version].append((int(net.network_address), int(net.broadcast_address)))
    result = {}
    for version, intervals in groups.items():
        merged = []
        for lo, hi in sorted(intervals):
            if merged and lo <= merged[-1][1] + 1:
                merged[-1] = (merged[-1][0], max(hi, merged[-1][1]))
            else: merged.append((lo, hi))
        result[version] = ([x[0] for x in merged], [x[1] for x in merged])
    return result

class IPPolicy:
    def __init__(self, allow, deny):
        self.allow, self.deny = compile_ranges(allow), compile_ranges(deny)

    @staticmethod
    def _contains(table, version, value):
        starts, ends = table[version]
        index = bisect_right(starts, value) - 1
        return index >= 0 and value <= ends[index]

    def is_allowed(self, text):
        try: address = ip_address(text)
        except ValueError: return False
        version, value = address.version, int(address)
        return (self._contains(self.allow, version, value)
                and not self._contains(self.deny, version, value))

p = IPPolicy(['10.0.0.0/8', '2001:db8::/32'], ['10.2.0.0/16'])
assert p.is_allowed('10.1.2.3') and not p.is_allowed('10.2.0.1')
assert p.is_allowed('2001:db8::1') and not p.is_allowed('192.0.2.1')
assert not p.is_allowed('bad') and not p.is_allowed('::ffff:10.1.2.3')
