# DHCP


# DHPC configuration

shared-network enp2s0f1 {
  subnet 10.10.0.0 netmask 255.255.0.0 {
    authoritative;
    max-lease-time 43200;
    min-lease-time 43200;
    default-lease-time 43200;
    option routers  10.10.0.1;
    next-server  10.10.0.1;
    option log-servers 10.10.0.1;
    option ntp-servers 10.10.0.1;
    option domain-name "entoto";
    option domain-name-servers  10.10.0.1;
    option domain-search  "entoto";
    range 10.10.1.11 10.10.1.21;
    option cumulus-provision-url "http://10.10.0.1:80/install/postscripts/cumulusztp";
    zone entoto. {
       primary 10.10.0.1; key xcat_key; 
    }
    zone 10.10.IN-ADDR.ARPA. {
       primary 10.10.0.1; key xcat_key; 
    }
    if option user-class-identifier = "xNBA" and option client-architecture = 00:00 { #x86, xCAT Network Boot Agent
        always-broadcast on;
        filename = "http://10.10.0.1:80/tftpboot/xcat/xnba/nets/10.10.0.0_16";
    } else if option user-class-identifier = "xNBA" and option client-architecture = 00:09 { #x86, xCAT Network Boot Agent
        filename = "http://10.10.0.1:80/tftpboot/xcat/xnba/nets/10.10.0.0_16.uefi";
    } else if option client-architecture = 00:00  { #x86
        filename "xcat/xnba.kpxe";
    } else if option vendor-class-identifier = "Etherboot-5.4"  { #x86
        filename "xcat/xnba.kpxe";
    } else if option client-architecture = 00:07 { #x86_64 uefi
         filename "xcat/xnba.efi";
    } else if option client-architecture = 00:09 { #x86_64 uefi alternative id
         filename "xcat/xnba.efi";
    } else if option client-architecture = 00:02 { #ia64
         filename "elilo.efi";
    } else if option client-architecture = 00:0e { #OPAL-v3
         option conf-file = "http://10.10.0.1:80/tftpboot/pxelinux.cfg/p/10.10.0.0_16";
    } else if substring (option vendor-class-identifier,0,11) = "onie_vendor" { #for onie on cumulus switch
        option www-server = "http://10.10.0.1:80/install/onie/onie-installer";
    } else if substring(filename,0,1) = null { #otherwise, provide yaboot if the client isn't specific
         filename "/yaboot";
    }
  } # 10.10.0.0/255.255.0.0 subnet_end
} # enp2s0f1 nic_end

# Routing table update
route -n 
route del -net 10.10.0.0 gw 0.0.0.0 netmask 255.255.0.0 dev enp2s0f1  # if local connection is found in the table


