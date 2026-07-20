package com.dtmarl.broker;

import org.cloudsimplus.brokers.DatacenterBroker;
import org.cloudsimplus.brokers.DatacenterBrokerSimple;
import org.cloudsimplus.core.CloudSimPlus;

public class BrokerManager {

    private final DatacenterBroker broker;

    public BrokerManager(CloudSimPlus simulation) {
        broker = new DatacenterBrokerSimple(simulation);
    }

    public DatacenterBroker getBroker() {
        return broker;
    }
}